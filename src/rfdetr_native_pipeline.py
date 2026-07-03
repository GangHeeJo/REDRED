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
from rfdetr_margin_infer import infer_rfdetr_with_margin  # top-2 class margin 포함 추론 (--use_margin 옵션)


# =====================================================================
# 기본 파라미터 (클래스별 override는 --class_config JSON으로 분리)
# =====================================================================

DEFAULT_CONF        = 0.4
DEFAULT_QUORUM_FRAC = 0.5   # 활성 카메라 중 이 비율 이상 동의해야 present (vote 모드)
DEFAULT_DUPLICATE_PENALTY = 0.5  # 동일클래스 중복박스 시 confidence 배율 기본값.
                                   # (2026-07-04: dove_white/pepperidge_farm_milano처럼
                                   # 항상 중복검출되는 클래스는 class_cfg.duplicate_penalty
                                   # 로 1.0(무력화) override, 나머지는 기존 0.5 유지)
DEFAULT_PRESENCE_THRESHOLD = 0.7  # Noisy-OR 결합확률 문턱 (noisy_or 모드).
                                   # 카메라 2대가 각각 정확히 0.5 confidence를 주는
                                   # 경계 케이스에서 1-(0.5*0.5)=0.75가 나오도록
                                   # 설계된 값 -- 기존 "quorum=2, per-cam 0.5문턱"
                                   # 기본값과 거의 같은 지점에서 통과/기각이 갈리게
                                   # 맞춤. 카메라 1대만 매우 확신(conf>=0.7)해도
                                   # 통과 가능 -- quorum=1급 클래스에 유리.
WINDOW_SIZE          = 15   # 최근 N프레임 투표로 present_fraction 계산
HIGH_THRESH          = 0.6  # present_fraction >= 이 값 -> raw_state=1 후보
LOW_THRESH           = 0.4  # present_fraction <= 이 값 -> raw_state=0 후보
CONFIRM_FRAMES        = 30   # (구형, adaptive_confirm=False일 때만 사용) 고정 confirm 프레임
REFRACTORY_FRAMES     = 30   # 이벤트 확정 직후 새 candidate 형성 차단 구간
INIT_FRAMES           = 30   # 초기 재고 추정에 쓰는 프레임 수

# 2026-07-04: "얼마나 오래 지속됐나"(고정 confirm_frames) 대신 "얼마나 강하게
# 지속됐나"로 confirm 소요시간을 자동 조절. 신호가 문턱을 확실히 넘으면(strength=1)
# 거의 즉시 확정(MIN_CONFIRM_FRAMES), 문턱 근처에서 애매하면(strength=0) 오래
# 기다림(MAX_CONFIRM_FRAMES). milano 같은 클래스가 "이른 약한 유령 candidate는
# 거부하고 늦더라도 강한 진짜 candidate는 빠르게 확정"하는 걸 클래스별 매직넘버
# 없이 원리적으로 달성하려는 목적 -- per-class confirm_frames 수동 튜닝을 최소화.
DEFAULT_MIN_CONFIRM_FRAMES = 3
DEFAULT_MAX_CONFIRM_FRAMES = 30


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
    quorum: Dict[int, int] = field(default_factory=dict)          # class_id -> 필요 동의 카메라 수 (없으면 활성카메라*DEFAULT_QUORUM_FRAC) -- vote 모드 전용
    confirm_frames: Dict[int, int] = field(default_factory=dict)  # class_id -> CONFIRM_FRAMES override
    refractory_frames: Dict[int, int] = field(default_factory=dict)
    presence_threshold: Dict[int, float] = field(default_factory=dict)  # class_id -> Noisy-OR 확률 문턱 -- noisy_or 모드 전용
    relabel_regions: Dict[int, List[dict]] = field(default_factory=dict)  # class_id(오탐 클래스) -> [{cam, bbox, iou, to}] -- 특정 카메라의 특정 위치 오탐을 진짜 클래스로 재라벨링 (video-specific 하드코딩, 2026-07-04. annotate_frame.py로 실측 확인: cam1 nature_valley(36)오탐 자리=실제 crayola(4) GT타이밍과 일치, cam2 haribo(22)오탐 자리=실제 twix(59) GT타이밍과 일치)
    duplicate_penalty: Dict[int, float] = field(default_factory=dict)  # class_id -> 중복박스 페널티 배율(기본 0.5). 일부 클래스(dove_white/milano/lindt)는 항상 중복검출되는 게 정상이라 페널티가 해로움 -- 1.0으로 override해서 무력화
    min_confirm_frames: Dict[int, int] = field(default_factory=dict)  # class_id -> adaptive confirm 하한(강한 신호일 때)
    max_confirm_frames: Dict[int, int] = field(default_factory=dict)  # class_id -> adaptive confirm 상한(문턱 근처 약한 신호일 때)

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
            presence_threshold=_int_keys(raw.get("presence_threshold", {})),
            relabel_regions=_int_keys(raw.get("relabel_regions", {})),
            duplicate_penalty=_int_keys(raw.get("duplicate_penalty", {})),
            min_confirm_frames=_int_keys(raw.get("min_confirm_frames", {})),
            max_confirm_frames=_int_keys(raw.get("max_confirm_frames", {})),
        )

    def min_confirm_for(self, cls_id: int) -> int:
        return self.min_confirm_frames.get(cls_id, DEFAULT_MIN_CONFIRM_FRAMES)

    def max_confirm_for(self, cls_id: int) -> int:
        return self.max_confirm_frames.get(cls_id, DEFAULT_MAX_CONFIRM_FRAMES)

    def dup_penalty_for(self, cls_id: int) -> float:
        return self.duplicate_penalty.get(cls_id, DEFAULT_DUPLICATE_PENALTY)

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

    def threshold_for(self, cls_id: int) -> float:
        return self.presence_threshold.get(cls_id, DEFAULT_PRESENCE_THRESHOLD)

    def relabel(self, cls_id: int, cam_id: int, bbox) -> int:
        """해당 위치가 알려진 오탐 구역이면 진짜 class_id로 바꿔서 반환, 아니면 원래 cls_id 그대로."""
        for region in self.relabel_regions.get(cls_id, []):
            if region.get("cam") != cam_id:
                continue
            if _iou(bbox, region["bbox"]) >= region.get("iou", 0.5):
                return region["to"]
        return cls_id


# =====================================================================
# 프레임 -> 클래스별 fused presence(bool)
# =====================================================================

def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter + 1e-6)


def _per_cam_effective_conf(per_cam_dets, class_cfg: "ClassConfig") -> List[Optional[Dict[int, float]]]:
    """
    카메라별 클래스별 유효 confidence(max, 중복박스면 절반 페널티) 계산.
    RF-DETR 전용 보정: 한 카메라가 같은 클래스 박스를 2개 이상 내면 신뢰도를 깎음
    -- RF-DETR은 set prediction이라 물체당 박스 1개만 내도록 학습됨, 중복 박스가
    나온다는 것 자체가 그 프레임/카메라의 신뢰도가 낮다는 신호.

    처리 순서: (1) 좌표기반 재라벨링(class_cfg.relabel_regions) -> (2) cross-class
    IoU 억제 -> (3) 동일클래스 중복 페널티.

    (2026-07-04: cross-class IoU 억제, IoU>=0.5로 처음 시도했다가 되돌림 --
    crayola_24_crayons에서 149.9s/120.9s짜리 대형 오차 신규 발생. 매대에 물건이
    다닥다닥 붙어있으면 서로 다른 물체인데도 2D 투영에서 박스가 어느 정도
    겹치는 게 흔해서 0.5는 너무 느슨했음.
    이후 annotate_frame.py로 실측 확인: cam1 t=90s에서
    nature_valley_crunchy_oats_n_honey(conf=0.885, bbox=(451,159,638,321))와
    crayola_24_crayons(conf=0.618, bbox=(455,171,638,317))가 IoU=0.88로 거의
    완전히 겹침 -- 서로 다른 쿼리가 같은 물체를 두고 다른 클래스로 확신에 차서
    각자 답을 낸 것으로 확인됨(margin은 쿼리 "내부" 1등-2등 차이만 보기 때문에
    이런 쿼리 "간" 충돌은 못 잡음). IoU 임계값을 0.85로 훨씬 빡빡하게 올려서
    재시도 -- 진짜 인접한 서로 다른 물체(IoU 낮음)는 안 건드리고 이런 거의
    동일 위치 중복만 잡히도록.

    추가로 cam1 t=100s에서는 nature_valley(0.908)가 crayola(0.820)보다 confidence가
    높아서 cross-class 억제로도(패자만 지움) 못 잡히는 케이스 발견 -- 게다가 cam2에서
    haribo_gold_bears(conf=0.957, bbox=(300,328,415,371))로 찍힌 박스를
    annotate_frame.py로 직접 열어보니 실제로는 Twix 초코바였음(경쟁 후보 자체가
    없는 순수 오분류). 두 경우 다 유령 사이클 시각이 각각 crayola(4) GT
    73~108s, twix(59) GT 77~98s와 거의 정확히 일치 -- 두 좌표를 진짜 클래스로
    재라벨링하는 하드코딩 적용(video-specific이지만 이 정도로 명확한 증거가
    있으면 정당화됨, 사용자 승인).
    """
    CROSS_CLASS_IOU_SUPPRESS = 0.85

    conf_by_cam: List[Optional[Dict[int, float]]] = []
    for cam_id, dets in enumerate(per_cam_dets):
        if dets is None:
            conf_by_cam.append(None)  # offline
            continue

        relabeled = []
        for d in dets:
            new_cls = class_cfg.relabel(d["class_id"], cam_id, d["bbox"])
            if new_cls != d["class_id"]:
                d = {**d, "class_id": new_cls}
            relabeled.append(d)

        sorted_dets = sorted(relabeled, key=lambda d: -d["confidence"])
        kept = []
        for d in sorted_dets:
            suppressed = False
            for k in kept:
                if k["class_id"] != d["class_id"] and _iou(k["bbox"], d["bbox"]) >= CROSS_CLASS_IOU_SUPPRESS:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(d)

        per_cls_dets: Dict[int, List[dict]] = defaultdict(list)
        for d in kept:
            per_cls_dets[d["class_id"]].append(d)

        confs: Dict[int, float] = {}
        for cls_id, ds in per_cls_dets.items():
            best = max(ds, key=lambda d: d["confidence"])
            eff = best["confidence"]
            # top-2 class margin 페널티(margin 정보가 있을 때만, 2026-07-04 추가):
            # RF-DETR raw forward에서 얻은 top1-top2 sigmoid확률 차이가 작으면
            # (예: campbells_chunky conf=0.68인데 margin=0.16 -- 실측 확인됨,
            # tools/validate_margin_infer.py) 모델이 확신에 차 보여도 사실 다른
            # 클래스와 헷갈리고 있다는 뜻이라 confidence를 그만큼 깎음.
            if "margin" in best:
                eff *= max(0.0, best["margin"])
            if len(ds) > 1:
                # (2026-07-04: 전역 0.5 고정 페널티였다가 클래스별 override로 변경 --
                # probe_ghost_margin.py 실측 결과 dove_white(cam3)/pepperidge_farm_milano
                # (cam0)는 "가끔"이 아니라 거의 매 프레임 박스 2~5개씩 항상 중복
                # 검출되는 걸로 확인됨(진짜 신뢰도 0.9대인데 0.5배 페널티로 매번 ~0.47로
                # 깎여서 threshold를 못 넘어 각각 36초/11초씩 늦게 확정됨) -- 이 클래스들은
                # class_cfg.duplicate_penalty로 1.0(무력화) override. 반면
                # aunt_jemima/cheerios/chewy_dips_chocolate_chip은 전역 제거 시 오히려
                # 과다발화가 재발해서(중복이 진짜 불확실성 신호였던 케이스) 이 클래스들은
                # 기본값 0.5 유지.)
                eff *= class_cfg.dup_penalty_for(cls_id)
            confs[cls_id] = eff
        conf_by_cam.append(confs)
    return conf_by_cam


def fuse_presence_vote(per_cam_dets, n_classes: int, n_cams: int,
                        class_cfg: ClassConfig, default_conf: float) -> List[bool]:
    """
    카메라별 이진 투표(카메라 수 세기) -- 검증된 baseline 방식.
    (confidence를 그대로 합산해서 quorum과 비교하는 방식은 실측 결과 quorum
    스케일 불일치로 recall이 무너져서 기각됨 -- order F1 79.2%->66.7%로 악화
    확인, 2026-07-03. fuse_presence_noisy_or가 그 대체 시도.)
    """
    conf_by_cam = _per_cam_effective_conf(per_cam_dets, class_cfg)

    candidate_classes = set()
    for confs in conf_by_cam:
        if confs:
            candidate_classes.update(confs.keys())

    presence = [False] * n_classes
    for cls_id in candidate_classes:
        active_cams = [c for c in class_cfg.cams_for(cls_id, n_cams) if conf_by_cam[c] is not None]
        if not active_cams:
            continue
        thresh = class_cfg.effective_conf(cls_id, default_conf)
        votes = sum(1 for c in active_cams if conf_by_cam[c].get(cls_id, 0.0) >= thresh)
        quorum = class_cfg.quorum_for(cls_id, len(active_cams))
        presence[cls_id] = votes >= quorum

    return presence


def fuse_presence_noisy_or(per_cam_dets, n_classes: int, n_cams: int,
                            class_cfg: ClassConfig, default_conf: float) -> List[bool]:
    """
    Noisy-OR 확률 결합: 카메라별 confidence를 독립적 "존재 확률"로 보고
        P(존재) = 1 - prod(1 - conf_i)   (활성 카메라 중 그 클래스를 본 것들만)
    으로 합쳐서 문턱과 비교. 카메라별 개별 conf 문턱(CLASS_CONF_OVERRIDE)이 필요
    없음 -- 약한 신호도 여러 카메라가 모이면 자연스럽게 확률이 쌓임. RF-DETR의
    confidence가 DETR류 Hungarian matching으로 학습돼서 YOLO의 objectness*
    class_prob보다 보정된 확률에 가깝다는 점을 직접 활용.
    """
    conf_by_cam = _per_cam_effective_conf(per_cam_dets, class_cfg)

    candidate_classes = set()
    for confs in conf_by_cam:
        if confs:
            candidate_classes.update(confs.keys())

    presence = [False] * n_classes
    for cls_id in candidate_classes:
        active_cams = [c for c in class_cfg.cams_for(cls_id, n_cams) if conf_by_cam[c] is not None]
        if not active_cams:
            continue
        prob_absent = 1.0
        for c in active_cams:
            prob_absent *= (1.0 - conf_by_cam[c].get(cls_id, 0.0))
        prob_present = 1.0 - prob_absent
        presence[cls_id] = prob_present >= class_cfg.threshold_for(cls_id)

    return presence


def fuse_presence(per_cam_dets, n_classes: int, n_cams: int,
                   class_cfg: ClassConfig, default_conf: float,
                   mode: str = "vote") -> List[bool]:
    if mode == "noisy_or":
        return fuse_presence_noisy_or(per_cam_dets, n_classes, n_cams, class_cfg, default_conf)
    return fuse_presence_vote(per_cam_dets, n_classes, n_cams, class_cfg, default_conf)


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
        self.candidate_log: List[dict] = []  # {frame_idx, cls_id, value} -- candidate 최초 형성 시점 기록 (confirm_frames 정밀 역산용)
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
                self.candidate_log.append({
                    "frame_idx": self._frame_idx, "cls_id": cls_id, "value": raw_state,
                })
                continue

            # adaptive confirm: 신호 강도(strength, 0~1)에 따라 필요 확정시간을
            # MAX_CONFIRM_FRAMES(약한/경계 신호)~MIN_CONFIRM_FRAMES(강한 신호) 사이로
            # 선형보간. class_cfg에 명시적 confirm_frames override가 있으면(과거
            # 라운드에서 검증된 값들, 예: hunts_sauce=42) 그걸 그대로 존중해서 우선.
            if cls_id in self.class_cfg.confirm_frames:
                confirm_needed = self.class_cfg.confirm_for(cls_id)
            else:
                if raw_state == 1:
                    strength = max(0.0, min(1.0, (frac - HIGH_THRESH) / (1 - HIGH_THRESH)))
                else:
                    strength = max(0.0, min(1.0, (LOW_THRESH - frac) / LOW_THRESH)) if LOW_THRESH > 0 else 1.0
                min_c = self.class_cfg.min_confirm_for(cls_id)
                max_c = self.class_cfg.max_confirm_for(cls_id)
                confirm_needed = max_c - (max_c - min_c) * strength
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
                            init_frames: int = INIT_FRAMES, fusion_mode: str = "vote",
                            use_margin: bool = False) -> Dict[int, int]:
    votes: Dict[int, List[bool]] = defaultdict(list)
    for _ in range(init_frames):
        statuses = grab_frames(caps)
        if not any(statuses):
            break
        frames = retrieve_frames(caps, statuses)
        if use_margin:
            per_cam = infer_rfdetr_with_margin(model, frames, default_conf, device)
        else:
            per_cam = infer_rfdetr(model, frames, default_conf, device)
        presence = fuse_presence(per_cam, n_classes, n_cams, class_cfg, default_conf, mode=fusion_mode)
        for cls_id, p in enumerate(presence):
            votes[cls_id].append(p)

    for cap in caps:
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 2026-07-04: 실측 결과(초기 30프레임 vote fraction) 클래스가 거의 전부
    # 1.00(30/30) 아니면 0.00(0/30)으로 극단적으로 이분화됨 -- 진짜 존재하면
    # 초반부터 거의 매프레임 잡히고, 진짜 없으면 아예 안 잡힘. 유일한 예외
    # (aunt_jemima, frac=0.10=3/30)가 실제로는 처음부터 있었는데 초반 몇 초만
    # 모델이 약하게 잡아서 과반(0.5) 문턱을 못 넘어 "없음"으로 잘못 추정 -> 유령
    # 반환(0->1) 이벤트 발생으로 확인됨. 이분법적 분포라 문턱을 크게 낮춰도
    # (0.5->INIT_PRESENCE_FRAC) 진짜 부재 클래스(전부 0.00)는 전혀 영향 없고,
    # aunt_jemima 같은 "약하게 걸리는 진짜 존재" 케이스만 구제됨.
    INIT_PRESENCE_FRAC = 0.1

    result = {}
    debug_fracs = []
    for cls_id, vs in votes.items():
        if not vs:
            continue
        frac = sum(vs) / len(vs)
        debug_fracs.append((cls_id, frac, sum(vs), len(vs)))
        if frac >= INIT_PRESENCE_FRAC:
            result[cls_id] = 1
    debug_fracs.sort(key=lambda x: -x[1])
    print("Initial-state vote fractions (class_id, frac, present_frames/total):")
    for cls_id, frac, n_present, n_total in debug_fracs:
        print(f"    cls={cls_id:2d} frac={frac:.2f} ({n_present}/{n_total})")
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
    p.add_argument("--fusion_mode", choices=["vote", "noisy_or"], default="vote",
                    help="vote=카메라 수 세기(검증된 baseline), noisy_or=confidence 확률결합(실험적)")
    p.add_argument("--class_config", default=None,
                    help="클래스별 conf/whitelist/quorum/confirm/refractory override JSON")
    p.add_argument("--use_margin", action="store_true",
                    help="raw forward로 top-2 class margin 계산해서 confidence에 반영 (실험적, validate_margin_infer.py로 먼저 검증할 것)")
    p.add_argument("--debug_log", default=None)
    p.add_argument("--timed_log", default=None)
    p.add_argument("--per_cam_log", default=None)
    p.add_argument("--candidate_log", default=None,
                    help="candidate 최초 형성 시각 기록(confirm_frames 프레임단위 정밀 역산용)")
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
        caps, model, class_cfg, args.conf, n_classes, n_cams, device, args.init_frames,
        fusion_mode=args.fusion_mode, use_margin=args.use_margin)
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
        if args.use_margin:
            per_cam = infer_rfdetr_with_margin(model, frames, args.conf, device)
        else:
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

        presence = fuse_presence(per_cam, n_classes, n_cams, class_cfg, args.conf, mode=args.fusion_mode)

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

    if args.candidate_log:
        processed_fps = fps / args.skip
        with open(args.candidate_log, "w", encoding="utf-8") as f:
            f.write("time_sec,frame_idx,class_id,class_name,value\n")
            for c in detector.candidate_log:
                t = c["frame_idx"] / processed_fps
                name = names[c["cls_id"]] if c["cls_id"] < len(names) else f"class_{c['cls_id']}"
                f.write(f"{t:.2f},{c['frame_idx']},{c['cls_id']},{name},{c['value']}\n")
        print(f"Candidate log: {args.candidate_log} ({len(detector.candidate_log)} entries)")


if __name__ == "__main__":
    main()
