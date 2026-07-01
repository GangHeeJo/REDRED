"""
RF-DETR 파이프라인 — run_pipeline.py에서 추론 부분만 교체

Usage:
    conda activate rfdetr
    python src/run_pipeline_rfdetr.py \
        --videos cam0.mp4 ... cam4.mp4 \
        --weights runs/rfdetr/checkpoint_best_total.pth \
        --names   data/names.txt \
        --prices  data/prices.csv \
        --out     output/submission_rfdetr.csv \
        --conf    0.4 --skip 2 --device 0
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from event_detector import EventDetector
from multi_view_fusion import fuse
from csv_generator import load_prices, events_to_csv
from tracker import MultiCameraTracker
from infer_rfdetr import load_rfdetr, infer_rfdetr
from run_pipeline import (
    load_names, open_videos, grab_frames, retrieve_frames,
    read_frames, video_duration,
    compute_per_class_cam_weights, _DEFAULT_CAM_WEIGHTS,
    estimate_initial_inventory,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos",    nargs="+", required=True)
    p.add_argument("--weights",   required=True)
    p.add_argument("--names",     required=True)
    p.add_argument("--prices",    required=True)
    p.add_argument("--out",       default="output/submission_rfdetr.csv")
    p.add_argument("--conf",      type=float, default=0.4)
    p.add_argument("--skip",      type=int,   default=2)
    p.add_argument("--init_frames", type=int, default=30)
    p.add_argument("--device",    default="0")
    p.add_argument("--per_cam_log", default=None)
    p.add_argument("--debug_log",   default=None)
    p.add_argument("--timed_log",   default=None)
    p.add_argument("--use_tracker", action="store_true")
    p.add_argument("--tracker_max_age", type=int, default=15)
    args = p.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    names  = load_names(args.names)
    prices = load_prices(args.prices)

    print("Loading RF-DETR...")
    model = load_rfdetr(args.weights, num_classes=len(names), device=device)

    caps = open_videos(args.videos)
    duration = video_duration(args.videos)

    # 초기 재고 추정
    print("Estimating initial inventory...")
    init_inv = {}
    detect_count = defaultdict(int)
    count_values  = defaultdict(list)
    for _ in range(args.init_frames):
        frames = read_frames(caps)
        if all(f is None for f in frames):
            break
        per_cam = infer_rfdetr(model, frames, args.conf, device)
        cam_w   = compute_per_class_cam_weights(per_cam)
        fused   = fuse(per_cam, cam_weights=cam_w)
        for cls_id, cnt in fused.items():
            if cnt > 0:
                detect_count[cls_id] += 1
                count_values[cls_id].append(cnt)
    for cap in caps:
        if cap: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for cls_id, n in detect_count.items():
        if n >= 1:
            med = int(np.median(count_values[cls_id]))
            if med > 0:
                init_inv[cls_id] = med

    detector = EventDetector(class_names=names, initial_counts=init_inv)
    tracker  = MultiCameraTracker(max_age=args.tracker_max_age) if args.use_tracker else None

    debug_f   = open(args.debug_log,   "w") if args.debug_log   else None
    timed_f   = open(args.timed_log,   "w") if args.timed_log   else None
    per_cam_f = open(args.per_cam_log, "w") if args.per_cam_log else None
    if debug_f:   debug_f.write("frame_idx,class_id,class_name,count\n")
    if timed_f:   timed_f.write("time_sec,class_name,action\n")
    if per_cam_f: per_cam_f.write("frame_idx,cam_id,class_id,class_name,count\n")

    frame_idx = 0
    fps       = 30.0
    t_start   = time.time()

    print("Running RF-DETR pipeline...")
    while True:
        statuses = grab_frames(caps)
        if not any(statuses):
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        frames = retrieve_frames(caps, statuses)

        if tracker:
            per_cam_raw = infer_rfdetr(model, frames, args.conf, device)
            per_cam     = [tracker.update(i, dets or []) for i, dets in enumerate(per_cam_raw)]
        else:
            per_cam = infer_rfdetr(model, frames, args.conf, device)

        if per_cam_f:
            for cam_i, dets in enumerate(per_cam):
                if not dets: continue
                cnt_map = defaultdict(int)
                for d in dets: cnt_map[d["class_id"]] += 1
                for cls_id, cnt in cnt_map.items():
                    per_cam_f.write(f"{frame_idx},{cam_i},{cls_id},{names[cls_id]},{cnt}\n")

        cam_weights = compute_per_class_cam_weights(per_cam)
        fused = fuse(per_cam, cam_weights=cam_weights)

        if debug_f:
            for cls_id, cnt in fused.items():
                if cnt > 0:
                    debug_f.write(f"{frame_idx},{cls_id},{names[cls_id]},{cnt}\n")

        # fused dict → flat detection list (EventDetector API)
        flat_dets = [
            {"class_id": cls_id, "confidence": 1.0, "bbox": []}
            for cls_id, cnt in fused.items()
            for _ in range(cnt)
        ]

        t_sec = frame_idx / fps
        new_events = detector.update(flat_dets)
        for ev in new_events:
            if timed_f:
                timed_f.write(f"{t_sec:.2f},{ev.class_name},{ev.action}\n")

        frame_idx += 1

    t_elapsed = time.time() - t_start
    rtf = t_elapsed / duration if duration > 0 else 0
    print(f"RTF = {rtf:.4f}  ({t_elapsed:.1f}s / {duration:.1f}s)")

    for f in [debug_f, timed_f, per_cam_f]:
        if f: f.close()
    for cap in caps:
        if cap: cap.release()

    os.makedirs(Path(args.out).parent, exist_ok=True)
    events_to_csv(
        events=detector.all_events,
        prices=prices,
        out_path=args.out,
        initial_inventory=init_inv,
        include_action=True,
        total_mode="inventory",
        encoding="utf-8-sig",
    )
    print(f"Submission: {args.out}  ({len(detector.all_events)} events)")


if __name__ == "__main__":
    main()
