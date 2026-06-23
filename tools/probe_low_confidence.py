"""
Probe whether the model emits ANY signal (even below the normal 0.4
confidence threshold) for specific classes that show zero detections in
the live pipeline. Runs NMS at a much lower threshold and logs the max
confidence seen per class per frame, across all cameras.

If confidence stays near-zero everywhere -> the model fundamentally
doesn't represent this class well (training/visual issue, not a
threshold/calibration issue).
If confidence sometimes spikes close to (but under) 0.4 -> lowering the
per-class threshold could recover real detections.

Usage (run on server, needs GPU):
    PYTHONPATH=~/yolov7 python tools/probe_low_confidence.py \
        --videos <cam0> <cam1> <cam2> <cam3> <cam4> \
        --weights ~/Dataset/yolov7_custom.pt \
        --names data/names.txt \
        --classes bumblebee_albacore dove_white dove_pink \
        --low_conf 0.05 \
        --skip 10 \
        --out output/low_conf_probe.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from run_pipeline import load_model, _preprocess_single, open_videos, read_frames, load_names  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--names", required=True)
    parser.add_argument("--classes", nargs="+", required=True,
                        help="class names to probe (must match names.txt entries)")
    parser.add_argument("--low_conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip", type=int, default=10,
                        help="process every Nth frame (this is a probe, not the real pipeline)")
    parser.add_argument("--out", default="output/low_conf_probe.csv")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    class_names = load_names(args.names)
    name_to_id = {n: i for i, n in enumerate(class_names)}
    target_ids = {}
    for n in args.classes:
        if n not in name_to_id:
            print(f"WARNING: '{n}' not in names.txt, skipping")
            continue
        target_ids[name_to_id[n]] = n

    print(f"Probing classes: {target_ids}")
    print("Loading model...")
    model, nms_fn = load_model(args.weights, device)

    caps = open_videos(args.videos)

    rows = []
    frame_idx = 0
    n_processed = 0

    while True:
        frames = read_frames(caps)
        if all(f is None for f in frames):
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        valid_idx = [i for i, f in enumerate(frames) if f is not None]
        if valid_idx:
            tensors = [_preprocess_single(frames[i], args.img_size) for i in valid_idx]
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                preds = model(batch)[0]
            preds = nms_fn(preds, args.low_conf, args.iou)

            for out_i, cam_i in enumerate(valid_idx):
                pred = preds[out_i]
                if pred is None or len(pred) == 0:
                    continue
                for *xyxy, conf, cls in pred.cpu().numpy():
                    cls = int(cls)
                    if cls in target_ids:
                        rows.append([frame_idx, cam_i, cls, target_ids[cls], float(conf)])

        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  ...frame {frame_idx} processed, {len(rows)} target-class detections so far", flush=True)

        frame_idx += 1

    for cap in caps:
        if cap:
            cap.release()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "cam_idx", "class_id", "class_name", "confidence"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows -> {args.out}")
    for cls_id, name in target_ids.items():
        confs = [r[4] for r in rows if r[2] == cls_id]
        if confs:
            print(f"  {name}: {len(confs)} detections, max_conf={max(confs):.3f}, "
                  f"mean_conf={sum(confs)/len(confs):.3f}")
        else:
            print(f"  {name}: ZERO detections even at conf>={args.low_conf}")


if __name__ == "__main__":
    main()
