"""
RTF (Real-Time Factor) 측정 스크립트.
RTF = 처리시간 / 영상길이  (낮을수록 좋음, 1.0 미만이면 실시간 처리 가능)

Usage:
    python measure_rtf.py \
        --videos cam0.mp4 cam1.mp4 \
        --weights ~/yolov7/runs/train/finetune_v1/weights/best.pt \
        --device 0 \
        --img_size 640 \
        --skip 1
"""

import argparse
import time
import sys
import cv2
import torch
import numpy as np
from pathlib import Path


def load_model(weights, device):
    sys.path.insert(0, str(Path(weights).parent.parent.parent))
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    model = attempt_load(weights, map_location=device)
    model.eval()
    return model, non_max_suppression


def video_duration_and_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps, int(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos",   nargs="+", required=True)
    parser.add_argument("--weights",  required=True)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--conf",     type=float, default=0.4)
    parser.add_argument("--skip",     type=int, default=1,
                        help="Process every Nth frame")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    model, nms_fn = load_model(args.weights, device)

    # Warmup
    dummy = torch.zeros(1, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)

    max_duration = max(video_duration_and_frames(v)[0] for v in args.videos)
    caps = [cv2.VideoCapture(v) for v in args.videos]

    print(f"Measuring RTF over {len(caps)} camera(s), skip={args.skip} ...")
    t0 = time.time()
    frame_idx = 0
    total_infer_frames = 0

    while True:
        frames = []
        any_valid = False
        for cap in caps:
            ret, f = cap.read()
            if ret:
                frames.append(f)
                any_valid = True
            else:
                frames.append(None)
        if not any_valid:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        for f in frames:
            if f is None:
                continue
            img = cv2.resize(f, (args.img_size, args.img_size))
            img = img[:, :, ::-1].transpose(2, 0, 1)
            tensor = torch.from_numpy(np.ascontiguousarray(img)).float().to(device) / 255.0
            tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                pred = model(tensor)[0]
            nms_fn(pred, args.conf, 0.45)
            total_infer_frames += 1

        frame_idx += 1

    for cap in caps:
        cap.release()

    proc_time = time.time() - t0
    rtf = proc_time / max_duration if max_duration > 0 else float("inf")

    print(f"\n{'='*40}")
    print(f"Video duration   : {max_duration:.2f}s")
    print(f"Processing time  : {proc_time:.2f}s")
    print(f"Inferred frames  : {total_infer_frames}")
    print(f"RTF              : {rtf:.4f}  {'✓ 실시간 가능' if rtf < 1.0 else '✗ 실시간 불가'}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
