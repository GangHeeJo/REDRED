"""
Event-Triggered Pipeline + ROI Crop (v3)

매 프레임 YOLO 대신:
  1. 프레임 차분으로 interaction 구간 감지 (YOLO 없이, 매우 빠름)
  2. 구간 직전 N프레임(before_buffer) vs 직후 N프레임(after_buffer) median 비교
  3. delta 있는 클래스만 이벤트

v3 핵심 설계:
  - trigger마다 자체 완결된 before/after 비교 (전역 상태 없음)
  - before_buffer: IDLE 중 최근 N_BEFORE 프레임 rolling buffer
  - after_buffer: SETTLING 중 수집된 안정 프레임
  - 양쪽 N프레임 median → 단일 프레임 YOLO 노이즈 제거
  - 같은 인터랙션에서 두 번째 trigger:
      before_buffer = 직전 trigger의 after 상태 (IDLE에서 갱신됨)
      after_buffer = 동일 상태 → delta=0 → 이벤트 없음 (자연 중복 차단)

Usage:
    python src/run_event_triggered.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
        --weights ~/runs/kd/yolo11m_kd_0630_0036/weights/best.pt \
        --names data/names.txt --prices data/prices.csv \
        --out output/submission_et.csv
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np
from collections import deque
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
DIFF_PIXEL_THRESH      = 25
DIFF_AREA_THRESH       = 0.012
SETTLE_FRAMES          = 25
MAX_INTERACTION_FRAMES = 300
ROI_PAD                = 0.20
ROI_MIN_AREA           = 0.02
ROI_MAX_AREA           = 0.80

N_BEFORE = 5   # IDLE 중 유지할 before rolling buffer 크기
N_AFTER  = 5   # SETTLING 중 사용할 after 프레임 수


class MultiCamDiffMonitor:
    """
    State machine: IDLE → ACTIVE → SETTLING → (trigger) → IDLE

    trigger 반환값: (before_sample, after_sample, rois)
      before_sample: IDLE 중 수집한 최근 N_BEFORE 프레임셋 리스트
      after_sample:  SETTLING 중 수집한 안정 프레임셋 리스트 (마지막 N_AFTER개)
    """

    def __init__(self, n_cams,
                 diff_area_thresh=DIFF_AREA_THRESH,
                 settle_frames=SETTLE_FRAMES,
                 n_before=N_BEFORE):
        self.n_cams           = n_cams
        self.diff_area_thresh = diff_area_thresh
        self.settle_frames    = settle_frames

        self.prev_grays    = [None] * n_cams
        self.state         = "IDLE"
        self.settle_count  = 0
        self.active_frames = 0
        self.accum_diff    = [None] * n_cams
        self.after_buffer  = []
        # rolling buffer of stable frames (updated only in IDLE)
        self.before_buffer = deque(maxlen=n_before)

    def _compute_roi(self, accum, img_h, img_w):
        if accum is None:
            return None
        mask = accum > DIFF_PIXEL_THRESH
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
        pad_y = int((rmax - rmin) * ROI_PAD)
        pad_x = int((cmax - cmin) * ROI_PAD)
        rmin = max(0, rmin - pad_y);  rmax = min(img_h, rmax + pad_y + 1)
        cmin = max(0, cmin - pad_x);  cmax = min(img_w, cmax + pad_x + 1)
        area_ratio = (rmax - rmin) * (cmax - cmin) / (img_h * img_w)
        if area_ratio < ROI_MIN_AREA or area_ratio > ROI_MAX_AREA:
            return None
        return (cmin, rmin, cmax, rmax)

    def update(self, frames):
        """
        Returns:
          None                              — 이벤트 없음
          (before_sample, after_sample, rois) — YOLO 비교용 프레임셋 + ROI
        """
        grays = []
        for f in frames:
            if f is None:
                grays.append(None)
            else:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                g = cv2.GaussianBlur(g, (5, 5), 0)
                grays.append(g)

        diffs, motions = [], []
        for i in range(self.n_cams):
            if grays[i] is not None and self.prev_grays[i] is not None:
                d = cv2.absdiff(grays[i], self.prev_grays[i])
                diffs.append(d)
                motions.append(float((d > DIFF_PIXEL_THRESH).mean()))
            else:
                diffs.append(None)
                motions.append(0.0)

        max_motion = max(motions) if motions else 0.0

        for i, g in enumerate(grays):
            if g is not None:
                self.prev_grays[i] = g

        trigger = None

        def _accumulate():
            for i, d in enumerate(diffs):
                if d is None:
                    continue
                if self.accum_diff[i] is None:
                    self.accum_diff[i] = d
                else:
                    self.accum_diff[i] = np.maximum(self.accum_diff[i], d)

        if self.state == "IDLE":
            # IDLE: before_buffer 갱신
            self.before_buffer.append([f.copy() if f is not None else None for f in frames])
            if max_motion > self.diff_area_thresh:
                self.state = "ACTIVE"
                self.active_frames = 1
                self.accum_diff = [None] * self.n_cams
                self.after_buffer = []
                _accumulate()

        elif self.state == "ACTIVE":
            self.active_frames += 1
            _accumulate()
            if max_motion <= self.diff_area_thresh:
                self.state = "SETTLING"
                self.settle_count = 1
                self.after_buffer = [[f.copy() if f is not None else None for f in frames]]
            elif self.active_frames > MAX_INTERACTION_FRAMES:
                self.state = "IDLE"
                self.active_frames = 0
                self.accum_diff = [None] * self.n_cams
                self.after_buffer = []

        elif self.state == "SETTLING":
            if max_motion > self.diff_area_thresh:
                self.state = "ACTIVE"
                self.settle_count = 0
                self.after_buffer = []
                _accumulate()
            else:
                self.settle_count += 1
                self.after_buffer.append([f.copy() if f is not None else None for f in frames])
                if self.settle_count >= self.settle_frames:
                    before_sample = list(self.before_buffer)
                    after_sample  = self.after_buffer[-N_AFTER:]
                    rois = []
                    for i, f in enumerate(frames):
                        if f is not None:
                            h, w = f.shape[:2]
                            rois.append(self._compute_roi(self.accum_diff[i], h, w))
                        else:
                            rois.append(None)
                    self.state         = "IDLE"
                    self.settle_count  = 0
                    self.active_frames = 0
                    self.accum_diff    = [None] * self.n_cams
                    self.after_buffer  = []
                    if before_sample:
                        trigger = (before_sample, after_sample, rois)

        return trigger


def crop_frames(frames, rois):
    cropped = []
    for f, roi in zip(frames, rois):
        if f is None or roi is None:
            cropped.append(f)
        else:
            x1, y1, x2, y2 = roi
            cropped.append(f[y1:y2, x1:x2])
    return cropped


def fuse_frames(model, nms_fn, frames, conf, iou, img_size, device,
                rois=None, quorum=2, min_corroborate=2):
    if rois and any(r is not None for r in rois):
        inference_frames = crop_frames(frames, rois)
    else:
        inference_frames = frames
    per_cam = infer_batch(model, nms_fn, inference_frames, conf, iou, img_size, device)
    cam_w   = compute_per_class_cam_weights(per_cam, min_corroborate)
    return fuse(per_cam, cam_weights=cam_w, quorum=quorum), per_cam


def fuse_frames_multi(model, nms_fn, frames_list, conf, iou, img_size, device,
                      rois=None, quorum=2, min_corroborate=2):
    """여러 프레임셋 YOLO 추론 후 class별 median count + consistency 반환.

    Returns:
        counts     : {cls_id: median_count}
        consistent : {cls_id: True if ≥(N-1) frames agree on the median}
    """
    all_counts = []
    for frames in frames_list:
        counts, _ = fuse_frames(model, nms_fn, frames, conf, iou, img_size, device,
                                rois, quorum, min_corroborate)
        all_counts.append(counts)
    if not all_counts:
        return {}, {}
    all_cls = set()
    for c in all_counts:
        all_cls.update(c.keys())
    median_counts = {}
    consistent    = {}
    n = len(all_counts)
    for cls_id in all_cls:
        vals   = [c.get(cls_id, 0) for c in all_counts]
        median = sorted(vals)[n // 2]
        median_counts[cls_id] = median
        # consistent if at most 1 frame disagrees with median
        consistent[cls_id] = vals.count(median) >= n - 1
    return median_counts, consistent


def make_events(before_counts, before_conf, after_counts, after_conf,
                class_names, counter, frame_idx):
    events = []
    all_cls = set(before_counts) | set(after_counts)
    for cls_id in all_cls:
        # Skip if either side's count is unreliable (YOLO flickering)
        if not before_conf.get(cls_id, True) or not after_conf.get(cls_id, True):
            continue
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
    parser.add_argument("--diff_thresh",     type=float, default=DIFF_AREA_THRESH)
    parser.add_argument("--settle",          type=int,   default=SETTLE_FRAMES)
    parser.add_argument("--n_before",        type=int,   default=N_BEFORE,
                        help="before median 프레임 수 (기본 5)")
    parser.add_argument("--n_after",         type=int,   default=N_AFTER,
                        help="after median 프레임 수 (기본 5)")
    parser.add_argument("--quorum",          type=int,   default=2)
    parser.add_argument("--min_corroborate", type=int,   default=2)
    parser.add_argument("--timed_log",       default=None)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    class_names = load_names(args.names)
    prices      = load_prices(args.prices)

    print("모델 로드 중...")
    model, nms_fn = load_model(args.weights, device)

    caps   = open_videos(args.videos)
    n_cams = len(caps)

    fps_cap = cv2.VideoCapture(args.videos[0])
    fps = fps_cap.get(cv2.CAP_PROP_FPS) or 30
    fps_cap.release()

    monitor = MultiCamDiffMonitor(n_cams,
                                  diff_area_thresh=args.diff_thresh,
                                  settle_frames=args.settle,
                                  n_before=args.n_before)

    all_events = []
    counter    = [0]
    frame_idx  = 0
    yolo_calls = 0
    t_start    = time.time()

    timed_writer = None
    timed_file   = None
    if args.timed_log:
        import csv as _csv
        timed_file   = open(args.timed_log, "w", newline="", encoding="utf-8")
        timed_writer = _csv.writer(timed_file)
        timed_writer.writerow(["time_sec", "class_name", "action"])

    print(f"처리 시작... diff_thresh={args.diff_thresh}  settle={args.settle}fr  "
          f"n_before={args.n_before}  n_after={args.n_after}\n")

    while True:
        frames = read_frames(caps)
        if all(f is None for f in frames):
            break

        result = monitor.update(frames)

        if result is not None:
            before_sample, after_sample, rois = result
            roi_used = sum(1 for r in rois if r is not None)

            before_counts, before_conf = fuse_frames_multi(
                model, nms_fn, before_sample,
                args.conf, args.iou, args.img_size, device,
                rois, args.quorum, args.min_corroborate,
            )
            after_counts, after_conf = fuse_frames_multi(
                model, nms_fn, after_sample,
                args.conf, args.iou, args.img_size, device,
                rois, args.quorum, args.min_corroborate,
            )
            yolo_calls += len(before_sample) + len(after_sample)

            new_events = make_events(before_counts, before_conf,
                                     after_counts, after_conf,
                                     class_names, counter, frame_idx)
            all_events.extend(new_events)

            for ev in new_events:
                t_sec = round(frame_idx / fps, 2)
                print(f"  [Frame {frame_idx} / {t_sec}s] {ev.class_name}: {ev.action} "
                      f"({ev.before}→{ev.after})  ROI {roi_used}/5캠")
                if timed_writer:
                    timed_writer.writerow([t_sec, ev.class_name, ev.action])

        frame_idx += 1

    if timed_file:
        timed_file.close()

    elapsed = time.time() - t_start
    dur     = video_duration(args.videos)
    rtf     = elapsed / dur if dur > 0 else 0

    expected = max(frame_idx // 2, 1)
    print(f"\n완료: {frame_idx}프레임 / YOLO 호출 {yolo_calls}회 "
          f"(skip=2 기준 {expected}회 대비 {yolo_calls/expected*100:.1f}%)")
    print(f"처리시간: {elapsed:.1f}s  영상길이: {dur:.1f}s  RTF: {rtf:.4f}")
    print(f"감지 이벤트: {len(all_events)}개\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    events_to_csv(
        events=all_events,
        prices=prices,
        out_path=args.out,
        include_action=True,
        total_mode="inventory",
        encoding="utf-8-sig",
    )
    print(f"저장: {args.out}")

    for cap in caps:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
