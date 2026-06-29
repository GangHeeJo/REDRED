"""
Event-Triggered Pipeline + ROI Crop

매 프레임 YOLO 대신:
  1. 프레임 차분으로 interaction 구간 감지 (YOLO 없이, 매우 빠름)
  2. 구간 직전 안정 프레임 (before) vs 직후 안정 프레임 (after) 에만 YOLO 실행
  3. before/after fused count 차이 = 이벤트

ROI Crop (논문 아이디어):
  - ACTIVE 구간 동안 누적 diff → 변화 영역(ROI) 자동 계산
  - before/after 를 ROI 영역만 crop → YOLO 에 넣음
  - 효과: RTF 추가 감소 (작은 이미지) + 가려진 상품 해상도 향상

장점:
  - YOLO 실행 횟수 대폭 감소 → RTF 개선
  - 정적 장면의 YOLO 노이즈 FP 원천 차단
  - 손이 선반에 없을 때는 아예 이벤트를 만들지 않음

Usage:
    python src/run_event_triggered.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
        --weights ~/yolov7/runs/train/exp/weights/best.pt \
        --names   data/names.txt \
        --prices  data/prices.csv \
        --out     output/submission_et.csv
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_pipeline import (
    load_model, infer_batch, open_videos, read_frames,
    video_duration, load_names, compute_per_class_cam_weights,
)
from multi_view_fusion import fuse
from event_detector import Event
from csv_generator import load_prices, events_to_csv


# ── 차분 파라미터 ─────────────────────────────────────────────────
DIFF_PIXEL_THRESH = 25     # absdiff 픽셀값 변화 감지 임계값
DIFF_AREA_THRESH  = 0.012  # 전체 픽셀의 N% 이상 변하면 interaction
SETTLE_FRAMES     = 25     # interaction 후 이 프레임만큼 조용해야 after 확정
MAX_INTERACTION_FRAMES = 300  # 이 이상 지속되면 강제 종료 (오감지 방지)
ROI_PAD           = 0.20   # ROI bounding box 패딩 비율 (상하좌우 20%)
ROI_MIN_AREA      = 0.02   # ROI < 전체의 N% → 노이즈로 판단, 풀 프레임 사용
ROI_MAX_AREA      = 0.80   # ROI > 전체의 N% → crop 의미 없음, 풀 프레임 사용


class MultiCamDiffMonitor:
    """
    5카메라 프레임 차분을 감시해 interaction 구간을 감지.

    States:
      IDLE      — 정적 장면. 안정 프레임 계속 저장.
      ACTIVE    — 하나 이상의 카메라에서 큰 움직임 감지 (손 들어옴).
      SETTLING  — 움직임이 다시 줄어드는 중. SETTLE_FRAMES 동안 조용하면 확정.
    """

    def __init__(self, n_cams,
                 diff_area_thresh=DIFF_AREA_THRESH,
                 settle_frames=SETTLE_FRAMES):
        self.n_cams          = n_cams
        self.diff_area_thresh = diff_area_thresh
        self.settle_frames   = settle_frames

        self.prev_grays      = [None] * n_cams
        self.state           = "IDLE"
        self.settle_count    = 0
        self.active_frames   = 0
        self.before_frames   = [None] * n_cams  # interaction 직전 안정 프레임
        self.accum_diff      = [None] * n_cams  # ACTIVE 동안 누적 diff (max)

    def _compute_roi(self, accum, img_h, img_w):
        """누적 diff → ROI (x1,y1,x2,y2). 변화 없거나 너무 작으면 None."""
        if accum is None:
            return None
        mask = accum > DIFF_PIXEL_THRESH
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
        # 패딩
        pad_y = int((rmax - rmin) * ROI_PAD)
        pad_x = int((cmax - cmin) * ROI_PAD)
        rmin = max(0, rmin - pad_y);  rmax = min(img_h, rmax + pad_y + 1)
        cmin = max(0, cmin - pad_x);  cmax = min(img_w, cmax + pad_x + 1)
        area_ratio = (rmax - rmin) * (cmax - cmin) / (img_h * img_w)
        if area_ratio < ROI_MIN_AREA or area_ratio > ROI_MAX_AREA:
            return None
        return (cmin, rmin, cmax, rmax)  # x1,y1,x2,y2

    def update(self, frames):
        """
        frames: list[np.ndarray | None] — 이번 프레임 (카메라 수만큼)
        Returns:
          None                              — 이벤트 없음
          (before, after, rois)             — YOLO 비교할 프레임 쌍 + 카메라별 ROI
        """
        grays = []
        for f in frames:
            if f is None:
                grays.append(None)
            else:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                g = cv2.GaussianBlur(g, (5, 5), 0)
                grays.append(g)

        # diff 계산 (prev_grays 업데이트 전에 해야 함)
        diffs = []
        motions = []
        for i in range(self.n_cams):
            if grays[i] is not None and self.prev_grays[i] is not None:
                d = cv2.absdiff(grays[i], self.prev_grays[i])
                diffs.append(d)
                motions.append(float((d > DIFF_PIXEL_THRESH).mean()))
            else:
                diffs.append(None)
                motions.append(0.0)

        max_motion = max(motions) if motions else 0.0

        # prev_grays 업데이트 (diff 계산 후)
        for i, g in enumerate(grays):
            if g is not None:
                self.prev_grays[i] = g

        trigger = None

        def _accumulate(diffs):
            for i, d in enumerate(diffs):
                if d is None:
                    continue
                if self.accum_diff[i] is None:
                    self.accum_diff[i] = d  # cv2.absdiff → already uint8
                else:
                    self.accum_diff[i] = np.maximum(self.accum_diff[i], d)

        if self.state == "IDLE":
            if max_motion > self.diff_area_thresh:
                self.state = "ACTIVE"
                self.active_frames = 1
                self.accum_diff = [None] * self.n_cams
                _accumulate(diffs)  # 전환 첫 프레임도 누적
            else:
                for i, f in enumerate(frames):
                    if f is not None:
                        self.before_frames[i] = f.copy()

        elif self.state == "ACTIVE":
            self.active_frames += 1
            _accumulate(diffs)
            if max_motion <= self.diff_area_thresh:
                self.state = "SETTLING"
                self.settle_count = 1  # 이 조용한 프레임 자체를 1로 계산 시작
            elif self.active_frames > MAX_INTERACTION_FRAMES:
                # 너무 오래 지속 → 강제 리셋
                self.state = "IDLE"
                self.active_frames = 0
                self.accum_diff = [None] * self.n_cams

        elif self.state == "SETTLING":
            if max_motion > self.diff_area_thresh:
                # 다시 움직임 → ACTIVE 복귀, 누적 이어서
                self.state = "ACTIVE"
                self.settle_count = 0
                _accumulate(diffs)
            else:
                self.settle_count += 1
                if self.settle_count >= self.settle_frames:
                    before = list(self.before_frames)
                    after  = [f.copy() if f is not None else None for f in frames]
                    rois = []
                    for i, f in enumerate(frames):
                        if f is not None:
                            h, w = f.shape[:2]
                            rois.append(self._compute_roi(self.accum_diff[i], h, w))
                        else:
                            rois.append(None)
                    for i, f in enumerate(frames):
                        if f is not None:
                            self.before_frames[i] = f.copy()
                    self.state        = "IDLE"
                    self.settle_count = 0
                    self.active_frames = 0
                    self.accum_diff   = [None] * self.n_cams
                    trigger = (before, after, rois)

        return trigger


def crop_frames(frames, rois):
    """각 카메라 프레임을 ROI 영역으로 crop. roi=None이면 전체 사용."""
    cropped = []
    for f, roi in zip(frames, rois):
        if f is None or roi is None:
            cropped.append(f)
        else:
            x1, y1, x2, y2 = roi
            cropped.append(f[y1:y2, x1:x2])
    return cropped


def fuse_frames(model, nms_fn, frames, conf, iou, img_size, device, rois=None):
    """프레임 배치 YOLO → 5카메라 fusion → {cls_id: count}"""
    # rois가 전부 None이면 풀 프레임 그대로 사용
    if rois and any(r is not None for r in rois):
        inference_frames = crop_frames(frames, rois)
    else:
        inference_frames = frames
    per_cam = infer_batch(model, nms_fn, inference_frames, conf, iou, img_size, device)
    cam_w   = compute_per_class_cam_weights(per_cam)
    return fuse(per_cam, cam_weights=cam_w), per_cam


def make_events(before_counts, after_counts, class_names, counter, frame_idx):
    """before/after count 비교 → Event 리스트."""
    events = []
    all_cls = set(before_counts) | set(after_counts)

    for cls_id in all_cls:
        b = before_counts.get(cls_id, 0)
        a = after_counts.get(cls_id, 0)
        delta = a - b
        if delta == 0 or not (1 <= abs(delta) <= 4):
            continue

        action = "반환" if delta > 0 else "구매"
        counter[0] += 1
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        events.append(Event(
            event_num  = counter[0],
            class_id   = cls_id,
            class_name = name,
            action     = action,
            before     = b,
            after      = a,
            frame_idx  = frame_idx,
        ))
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos",    nargs="+", required=True)
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--names",     required=True)
    parser.add_argument("--prices",    required=True)
    parser.add_argument("--out",       default="output/submission_et.csv")
    parser.add_argument("--conf",      type=float, default=0.4)
    parser.add_argument("--iou",       type=float, default=0.45)
    parser.add_argument("--img_size",  type=int,   default=640)
    parser.add_argument("--device",    default="0")
    parser.add_argument("--diff_thresh", type=float, default=DIFF_AREA_THRESH,
                        help="프레임 차분 임계값 (기본 0.012)")
    parser.add_argument("--settle",    type=int,   default=SETTLE_FRAMES,
                        help="안정화 필요 프레임 수 (기본 25)")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    class_names = load_names(args.names)
    prices      = load_prices(args.prices, class_names)

    print("모델 로드 중...")
    model, nms_fn = load_model(args.weights, device)

    caps = open_videos(args.videos)
    n_cams = len(caps)
    monitor = MultiCamDiffMonitor(n_cams,
                                  diff_area_thresh=args.diff_thresh,
                                  settle_frames=args.settle)

    all_events = []
    counter    = [0]   # mutable int
    frame_idx  = 0
    yolo_calls = 0
    t_start    = time.time()

    print("처리 시작... (차분 감시 중)")
    print(f"  diff_thresh={args.diff_thresh}  settle={args.settle}프레임\n")

    while True:
        frames = read_frames(caps)
        if all(f is None for f in frames):
            break

        result = monitor.update(frames)

        if result is not None:
            before_frames, after_frames, rois = result

            # before_frames 가 전부 None이면 (영상 시작 직후 trigger) 건너뜀
            if all(f is None for f in before_frames):
                frame_idx += 1
                continue

            yolo_calls += 2
            roi_used = sum(1 for r in rois if r is not None)

            b_counts, _ = fuse_frames(model, nms_fn, before_frames,
                                      args.conf, args.iou, args.img_size, device, rois)
            a_counts, _ = fuse_frames(model, nms_fn, after_frames,
                                      args.conf, args.iou, args.img_size, device, rois)

            new_events = make_events(b_counts, a_counts, class_names,
                                     counter, frame_idx)
            all_events.extend(new_events)

            for ev in new_events:
                print(f"  [Frame {frame_idx}] {ev.class_name}: {ev.action} "
                      f"({ev.before}->{ev.after})  ROI {roi_used}/5캠")

        frame_idx += 1

    elapsed = time.time() - t_start
    dur     = video_duration(args.videos)
    rtf     = elapsed / dur if dur > 0 else 0

    print(f"\n완료: {frame_idx}프레임 / YOLO 호출 {yolo_calls}회 "
          f"(기존 약 {frame_idx // 2}회 대비 {yolo_calls / max(frame_idx//2,1)*100:.1f}%)")
    print(f"처리시간: {elapsed:.1f}s  영상길이: {dur:.1f}s  RTF: {rtf:.4f}")
    print(f"감지 이벤트: {len(all_events)}개\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    events_to_csv(all_events, prices, args.out)
    print(f"저장: {args.out}")

    for cap in caps:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
