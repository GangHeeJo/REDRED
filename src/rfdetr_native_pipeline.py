"""
RF-DETR 전용 파이프라인 — 처음부터 새로 작성 (2026-07-03).

기존 run_pipeline.py/event_detector.py/multi_view_fusion.py/tracker.py/
csv_generator.py를 전혀 참조하지 않음. YOLOv7 파이프라인에서 발견된
사실(GT 재고는 모든 클래스에서 항상 0 또는 1, 절대 2 이상 동시존재하지 않음
-- data/ground_truth_v2.csv 전체 검증됨)만 설계에 반영하고, 로직은 완전히 새로
짬.

핵심 설계:
  1. 재고는 클래스당 항상 0/1 (binary presence) -- count 개념 자체를 없앰.
  2. 카메라별 감지를 프레임마다 "이 클래스가 보이는가"(bool)로 축약.
  3. 여러 카메라 투표 + 클래스별 quorum/whitelist/conf로 프레임별 fused presence 결정.
  4. hysteresis band(고/저 임계값)로 프레임 단위 노이즈를 1차로 걸러낸 뒤,
     raw_state가 committed와 달라지면 candidate로 넣고 CONFIRM_FRAMES 연속
     유지돼야 확정 (진짜 상태변화만 이벤트로 인정).
  5. 이벤트 확정 직후 REFRACTORY_FRAMES 동안은 새 candidate 형성 자체를 차단
     (짧은 occlusion 잔향으로 인한 재발화 방지) -- 기존 코드의 "history.clear()
     후 window 재적립"보다 명시적이고 강한 차단.
  6. 클래스별 quorum/whitelist/conf/confirm/refractory는 전부 JSON 설정파일로
     분리 (--class_config) -- 코드 수정 없이 튜닝 가능.

Usage:
    conda activate rfdetr
    python src/rfdetr_native_pipeline.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
        --weights runs/rfdetr/checkpoint_best_total.pth \
        --names data/names.txt --prices data/prices.csv \
        --out output/submission_native.csv \
        --skip 3 --conf 0.35 --device 0
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from infer_rfdetr import load_rfdetr, infer_rfdetr  # RF-DETR 전용 모델 래퍼, 순수 추론만 함 -- 재사용


# =====================================================================
# 기본 파라미터 (클래스별 override는 --class_config JSON으로 분리)
# =====================================================================

DEFAULT_CONF        = 0.4
DEFAULT_QUORUM_FRAC = 0.5   # 활성 카메라 중 이 비율 이상 동의해야 present
WINDOW_SIZE          = 15   # 최근 N프레임 투표로 present_fraction 계산
HIGH_THRESH          = 0.6  # present_fraction >= 이 값 -> raw_state=1 후보
LOW_THRESH           = 0.4  # present_fraction <= 이 값 -> raw_state=0 후보
CONFIRM_FRAMES        = 30   # candidate가 이 프레임수만큼 연속 유지돼야 이벤트 확정
REFRACTORY_FRAMES     = 30   # 이벤트 확정 직후 새 candidate 형성 차단 구간
INIT_FRAMES           = 30   # 초기 재고 추정에 쓰는 프레임 수


# =====================================================================
# 비디오 입출력 (순수 OpenCV, 처음부터 작성)
# =====================================================================

def open_videos(paths: List[str]):
    caps = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        caps.append(cap if cap.isOpened() else None)
    return caps


def grab_frames(caps) -> List[bool]:
    return [cap.grab() if cap is not None else False for cap in caps]


def retrieve_frames(caps, statuses: List[bool]):
    frames = []
    for cap, ok in zip(caps, statuses):
        if cap is not None and ok:
            ret, frame = cap.retrieve()
            frames.append(frame if ret else None)
        else:
            frames.append(None)
    return frames


def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n_frames / fps if fps else 0.0


# =====================================================================
# 이름/가격 로딩 (순수 csv, 처음부터 작성)
# =====================================================================

def load_names(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_prices(path: str) -> Dict[int, Decimal]:
    """class_id -> price(원). prices.csv: class_id,class_name,price_krw"""
    prices = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cls_id = int(row["class_id"])
            price_field = row.get("price_krw") or row.get("price") or "0"
            prices[cls_id] = Decimal(str(price_field).replace(",", "").strip())
    return prices


# =====================================================================
# 클래스별 설정 (JSON 파일로 분리 -- 코드 수정 없이 튜닝)
# =====================================================================

@dataclass
class ClassConfig:
    conf: Dict[int, float] = field(default_factory=dict)          # class_id -> effective conf threshold
    whitelist: Dict[int, List[int]] = field(default_factory=dict) # class_id -> [cam_ids] (없으면 전체 카메라 사용)
    quorum: Dict[int, int] = field(default_factory=dict)          # class_id -> 필요 동의 카메라 수 (없으면 활성카메라*DEFAULT_QUORUM_FRAC)
    confirm_frames: Dict[int, int] = field(default_factory=dict)  # class_id -> CONFIRM_FRAMES override
    refractory_frames: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str]) -> "ClassConfig":
        if not path or not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        def _int_keys(d):
            return {int(k): v for k, v in d.items()}
        return cls(
            conf=_int_keys(raw.get("conf", {})),
            whitelist=_int_keys(raw.get("whitelist", {})),
            quorum=_int_keys(raw.get("quorum", {})),
            confirm_frames=_int_keys(raw.get("confirm_frames", {})),
            refractory_frames=_int_keys(raw.get("refractory_frames", {})),
        )

    def effective_conf(self, cls_id: int, default: float) -> float:
        return self.conf.get(cls_id, default)

    def cams_for(self, cls_id: int, n_cams: int) -> List[int]:
        return self.whitelist.get(cls_id, list(range(n_cams)))

    def quorum_for(self, cls_id: int, n_active_cams: int) -> int:
        if cls_id in self.quorum:
            return self.quorum[cls_id]
        return max(1, round(n_active_cams * DEFAULT_QUORUM_FRAC))

    def confirm_for(self, cls_id: int) -> int:
        return self.confirm_frames.get(cls_id, CONFIRM_FRAMES)

    def refractory_for(self, cls_id: int) -> int:
        return self.refractory_frames.get(cls_id, REFRACTORY_FRAMES)


# =====================================================================
# 프레임 -> 클래스별 fused presence(bool)
# =====================================================================

def fuse_presence(per_cam_dets, n_classes: int, n_cams: int,
                   class_cfg: ClassConfig, default_conf: float) -> List[bool]:
    """
    per_cam_dets: [[{class_id, confidence, bbox}, ...] or None, ...]  (카메라당 1개)
    반환: 길이 n_classes인 bool 리스트 (fused presence)
    """
    # 카메라별로 conf 필터 후 "이 카메라가 본 클래스 집합" 계산
    seen_by_cam: List[set] = []
    for dets in per_cam_dets:
        if dets is None:
            seen_by_cam.append(None)  # offline
            continue
        s = set()
        for d in dets:
            thresh = class_cfg.effective_conf(d["class_id"], default_conf)
            if d["confidence"] >= thresh:
                s.add(d["class_id"])
        seen_by_cam.append(s)

    candidate_classes = set()
    for s in seen_by_cam:
        if s:
            candidate_classes.update(s)

    presence = [False] * n_classes
    for cls_id in candidate_classes:
        active_cams = [c for c in class_cfg.cams_for(cls_id, n_cams) if seen_by_cam[c] is not None]
        if not active_cams:
            continue
        votes = sum(1 for c in active_cams if cls_id in seen_by_cam[c])
        quorum = class_cfg.quorum_for(cls_id, len(active_cams))
        presence[cls_id] = votes >= quorum

    return presence


# =====================================================================
# 클래스별 hysteresis + candidate/confirm/refractory 상태머신
# =====================================================================

@dataclass
class ClassState:
    history: deque
    committed: int = 0          # 0/1
    candidate: Optional[int] = None
    candidate_since: int = 0
    refractory_until: int = -1


@dataclass
class Event:
    event_num: int
    class_id: int
    class_name: str
    action: str        # "구매" | "반환"
    before: int
    after: int
    frame_idx: int


class PresenceEventDetector:
    def __init__(self, class_names: List[str], class_cfg: ClassConfig,
                 initial_state: Optional[Dict[int, int]] = None):
        self.class_names = class_names
        self.class_cfg = class_cfg
        self.states: Dict[int, ClassState] = {}
        self.all_events: List[Event] = []
        self._event_counter = 0
        self._frame_idx = 0
        initial_state = initial_state or {}
        for cls_id, val in initial_state.items():
            self._state(cls_id).committed = val

    def _state(self, cls_id: int) -> ClassState:
        if cls_id not in self.states:
            self.states[cls_id] = ClassState(history=deque(maxlen=WINDOW_SIZE))
        return self.states[cls_id]

    def update(self, presence: List[bool]) -> List[Event]:
        new_events = []
        touched = set(i for i, p in enumerate(presence) if p) | set(self.states.keys())

        for cls_id in touched:
            st = self._state(cls_id)
            st.history.append(1 if (cls_id < len(presence) and presence[cls_id]) else 0)

            if len(st.history) < WINDOW_SIZE:
                continue

            frac = sum(st.history) / len(st.history)
            if frac >= HIGH_THRESH:
                raw_state = 1
            elif frac <= LOW_THRESH:
                raw_state = 0
            else:
                raw_state = st.committed  # 애매구간: 변화 없음으로 취급

            # 리프랙토리 구간이면 candidate 형성 자체를 막음
            if self._frame_idx < st.refractory_until:
                st.candidate = None
                continue

            if raw_state == st.committed:
                st.candidate = None
                continue

            if st.candidate != raw_state:
                st.candidate = raw_state
                st.candidate_since = self._frame_idx
                continue

            confirm_needed = self.class_cfg.confirm_for(cls_id)
            if self._frame_idx - st.candidate_since >= confirm_needed:
                before = st.committed
                after = raw_state
                action = "반환" if after > before else "구매"

                self._event_counter += 1
                ev = Event(
                    event_num=self._event_counter,
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    action=action,
                    before=before,
                    after=after,
                    frame_idx=self._frame_idx,
                )
                st.committed = after
                st.candidate = None
                st.refractory_until = self._frame_idx + self.class_cfg.refractory_for(cls_id)
                st.history.clear()
                self.all_events.append(ev)
                new_events.append(ev)

        self._frame_idx += 1
        return new_events


# =====================================================================
# 초기 재고 추정 (binary majority vote)
# =====================================================================

def estimate_initial_state(caps, model, class_cfg: ClassConfig, default_conf: float,
                            n_classes: int, n_cams: int, device: str,
                            init_frames: int = INIT_FRAMES) -> Dict[int, int]:
    votes: Dict[int, List[bool]] = defaultdict(list)
    for _ in range(init_frames):
        statuses = grab_frames(caps)
        if not any(statuses):
            break
        frames = retrieve_frames(caps, statuses)
        per_cam = infer_rfdetr(model, frames, default_conf, device)
        presence = fuse_presence(per_cam, n_classes, n_cams, class_cfg, default_conf)
        for cls_id, p in enumerate(presence):
            votes[cls_id].append(p)

    for cap in caps:
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    result = {}
    for cls_id, vs in votes.items():
        if not vs:
            continue
        frac = sum(vs) / len(vs)
        if frac >= 0.5:
            result[cls_id] = 1
    return result


# =====================================================================
# CSV 출력 (제출 포맷 -- 기존 csv_generator.py와 동일한 컬럼, 새로 작성)
# =====================================================================

def write_submission_csv(events: List[Event], prices: Dict[int, Decimal],
                          names: List[str], initial_state: Dict[int, int],
                          out_path: str):
    inventory = dict(initial_state)
    header = ["품목명", "이벤트 번호", "구매/반환 여부", "이벤트 후 재고 수량", "총 재고 금액"]
    rows = []
    for ev in sorted(events, key=lambda e: e.event_num):
        inventory[ev.class_id] = ev.after
        total = sum(Decimal(inventory.get(c, 0)) * prices.get(c, Decimal(0))
                    for c in range(len(names)))
        total = total.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        rows.append([
            names[ev.class_id] if ev.class_id < len(names) else f"class_{ev.class_id}",
            f"Event {ev.event_num}",
            ev.action,
            f"{ev.after}개",
            f"{total}원",
        ])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Submission: {out_path} ({len(rows)} events)")


# =====================================================================
# 메인
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--prices", required=True)
    p.add_argument("--out", default="output/submission_native.csv")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--skip", type=int, default=3)
    p.add_argument("--init_frames", type=int, default=INIT_FRAMES)
    p.add_argument("--device", default="0")
    p.add_argument("--class_config", default=None,
                    help="클래스별 conf/whitelist/quorum/confirm/refractory override JSON")
    p.add_argument("--debug_log", default=None)
    p.add_argument("--timed_log", default=None)
    p.add_argument("--per_cam_log", default=None)
    args = p.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    names = load_names(args.names)
    prices = load_prices(args.prices)
    class_cfg = ClassConfig.load(args.class_config)
    n_classes = len(names)

    print("Loading RF-DETR...")
    model = load_rfdetr(args.weights, num_classes=n_classes, device=device)

    caps = open_videos(args.videos)
    n_cams = len(caps)
    duration = video_duration(args.videos[0])

    print("Estimating initial state (binary presence)...")
    init_state = estimate_initial_state(
        caps, model, class_cfg, args.conf, n_classes, n_cams, device, args.init_frames)
    print(f"Initial state: {len(init_state)} classes present -- "
          f"{[names[c] for c in init_state]}")

    detector = PresenceEventDetector(names, class_cfg, initial_state=init_state)

    debug_f   = open(args.debug_log, "w") if args.debug_log else None
    timed_f   = open(args.timed_log, "w") if args.timed_log else None
    per_cam_f = open(args.per_cam_log, "w") if args.per_cam_log else None
    if debug_f:   debug_f.write("frame_idx,class_id,class_name,count\n")
    if timed_f:   timed_f.write("time_sec,class_name,action\n")
    if per_cam_f: per_cam_f.write("frame_idx,cam_id,class_id,class_name,count\n")

    frame_idx = 0
    fps = 30.0
    t_start = time.time()
    total_frames = int(cv2.VideoCapture(args.videos[0]).get(cv2.CAP_PROP_FRAME_COUNT))

    print("Running RF-DETR native pipeline...")
    while True:
        statuses = grab_frames(caps)
        if not any(statuses):
            break
        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        frames = retrieve_frames(caps, statuses)
        per_cam = infer_rfdetr(model, frames, args.conf, device)

        if per_cam_f:
            for cam_i, dets in enumerate(per_cam):
                if not dets:
                    continue
                cnt_map = defaultdict(int)
                for d in dets:
                    cnt_map[d["class_id"]] += 1
                for cls_id, cnt in cnt_map.items():
                    per_cam_f.write(f"{frame_idx},{cam_i},{cls_id},{names[cls_id]},{cnt}\n")

        presence = fuse_presence(per_cam, n_classes, n_cams, class_cfg, args.conf)

        if debug_f:
            for cls_id, p in enumerate(presence):
                if p:
                    debug_f.write(f"{frame_idx},{cls_id},{names[cls_id]},1\n")

        t_sec = frame_idx / fps
        new_events = detector.update(presence)
        for ev in new_events:
            if timed_f:
                timed_f.write(f"{t_sec:.2f},{ev.class_name},{ev.action}\n")
            print(f"  [{t_sec:6.1f}s] {ev.class_name}: {ev.action} ({ev.before}->{ev.after})")

        if frame_idx % 300 == 0:
            elapsed = time.time() - t_start
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            eta = elapsed / frame_idx * (total_frames - frame_idx) if frame_idx > 0 else 0
            print(f"[{pct:5.1f}%] frame {frame_idx}/{total_frames}  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  events {len(detector.all_events)}")

        frame_idx += 1

    t_elapsed = time.time() - t_start
    rtf = t_elapsed / duration if duration > 0 else 0
    print(f"RTF = {rtf:.4f}  ({t_elapsed:.1f}s / {duration:.1f}s)")

    for f in [debug_f, timed_f, per_cam_f]:
        if f:
            f.close()
    for cap in caps:
        if cap:
            cap.release()

    write_submission_csv(detector.all_events, prices, names, init_state, args.out)


if __name__ == "__main__":
    main()
