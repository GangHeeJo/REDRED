"""
Per-frame inventory analyzer: raw detection counts + change visualization.

Usage:
    python tools/analyze_inventory.py \
        --videos ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4 \
        --weights ~/Dataset/yolov7_custom.pt \
        --names   data/names.txt \
        --out     output/analysis \
        --skip    2
"""

import argparse
import sys
import csv
import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from run_pipeline import load_model, load_names, infer_batch, open_videos, grab_frames, retrieve_frames
from multi_view_fusion import fuse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos",   nargs="+", required=True)
    parser.add_argument("--weights",  required=True)
    parser.add_argument("--names",    required=True)
    parser.add_argument("--out",      default="output/analysis")
    parser.add_argument("--conf",     type=float, default=0.4)
    parser.add_argument("--iou",      type=float, default=0.45)
    parser.add_argument("--img_size", type=int,   default=640)
    parser.add_argument("--skip",     type=int,   default=2)
    parser.add_argument("--device",   default="0")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    print("Loading model...")
    model, nms_fn = load_model(args.weights, device)
    class_names = load_names(args.names)

    caps = open_videos(args.videos)
    frame_idx = 0

    # per-frame raw counts: list of {frame, class_id, class_name, count}
    rows = []

    print("Running inference...")
    while True:
        statuses = grab_frames(caps)
        if not any(statuses):
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        frames = retrieve_frames(caps, statuses)
        per_cam = infer_batch(model, nms_fn, frames, args.conf, args.iou, args.img_size, device)
        fused = fuse(per_cam)

        for cls_id, cnt in fused.items():
            rows.append({
                "frame":      frame_idx,
                "class_id":   cls_id,
                "class_name": class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}",
                "count":      cnt,
            })

        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx}...")

        frame_idx += 1

    for cap in caps:
        if cap:
            cap.release()

    # Save raw CSV
    raw_csv = os.path.join(args.out, "raw_counts.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "class_id", "class_name", "count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Raw counts saved: {raw_csv}")

    # Build per-class time series and detect raw events (no gap filter)
    class_series = defaultdict(dict)  # class_id -> {frame: count}
    for r in rows:
        class_series[r["class_id"]][r["frame"]] = r["count"]

    # Find classes with any change
    changed_classes = []
    for cls_id, series in class_series.items():
        counts = list(series.values())
        if max(counts) != min(counts):
            changed_classes.append(cls_id)

    print(f"\nClasses with inventory changes: {len(changed_classes)}")

    # Save events CSV (raw, no MIN_EVENT_GAP)
    events_csv = os.path.join(args.out, "raw_events.csv")
    all_frames = sorted(set(r["frame"] for r in rows))

    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "class_name", "action", "before", "after"])

        for cls_id in sorted(changed_classes):
            series = class_series[cls_id]
            name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            prev = None
            for fr in sorted(series.keys()):
                cnt = series[fr]
                if prev is not None and cnt != prev:
                    action = "구매" if cnt < prev else "반환"
                    writer.writerow([fr, name, action, prev, cnt])
                prev = cnt

    print(f"Raw events saved: {events_csv}")

    # Try to plot with matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot top 10 most-changing classes
        change_counts = {}
        for cls_id in changed_classes:
            series = class_series[cls_id]
            counts = list(series.values())
            change_counts[cls_id] = max(counts) - min(counts) + len(set(counts))

        top_classes = sorted(change_counts, key=change_counts.get, reverse=True)[:10]

        fig, axes = plt.subplots(len(top_classes), 1, figsize=(16, 2.5 * len(top_classes)))
        if len(top_classes) == 1:
            axes = [axes]

        for ax, cls_id in zip(axes, top_classes):
            series = class_series[cls_id]
            frames = sorted(series.keys())
            counts = [series[f] for f in frames]
            name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            ax.step(frames, counts, where="post", linewidth=1.5)
            ax.set_ylabel("count", fontsize=8)
            ax.set_title(name, fontsize=9)
            ax.set_xlim(min(all_frames), max(all_frames))
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("frame")
        plt.tight_layout()
        plot_path = os.path.join(args.out, "inventory_plot.png")
        plt.savefig(plot_path, dpi=120)
        print(f"Plot saved: {plot_path}")

    except ImportError:
        print("matplotlib not available, skipping plot")

    print("\nDone. Files in:", args.out)
    print("  raw_counts.csv  — 프레임별 전체 감지 수")
    print("  raw_events.csv  — 필터링 없는 raw 이벤트 목록")
    print("  inventory_plot.png — 재고 변화 그래프 (변화 많은 상위 10개 클래스)")


if __name__ == "__main__":
    main()
