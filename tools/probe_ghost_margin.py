"""
특정 클래스들의 유령 사이클(GT랑 안 맞는 발화) 구간에서 confidence+margin을
프레임별로 찍어서, 진짜 클래스 혼동(margin 낮음)인지 순수 occlusion 노이즈
(margin 높음, 다른 원인)인지 확정.

Usage:
    python tools/probe_ghost_margin.py \
        --weights runs/rfdetr/checkpoint_best_total.pth \
        --focus 36 22 6 \
        --start_sec 0 --end_sec 240 --step_sec 1
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from infer_rfdetr import load_rfdetr
from rfdetr_margin_infer import infer_rfdetr_with_margin


def load_names(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--names", default="data/names.txt")
    p.add_argument("--focus", nargs="+", type=int, required=True)
    p.add_argument("--start_sec", type=float, default=0)
    p.add_argument("--end_sec", type=float, default=240)
    p.add_argument("--step_sec", type=float, default=1.0)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default=None, help="지정하면 stdout 대신(+동시에) 이 파일에도 저장")
    args = p.parse_args()

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    def emit(line):
        print(line)
        if out_f:
            out_f.write(line + "\n")

    names = load_names(args.names)
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    cam_dir = Path.home() / "Dataset/4.TestVideo_Sample"

    print("모델 로딩...")
    model = load_rfdetr(args.weights, num_classes=len(names), device=device)

    caps = [cv2.VideoCapture(str(cam_dir / f"cam{i}/Sample_1.mp4")) for i in range(5)]
    fps = 30.0

    t = args.start_sec
    emit(f"{'time':>7} {'cam':>3} {'class':<40} {'conf':>6} {'margin':>7}")
    while t <= args.end_sec:
        frame_no = int(t * fps)
        frames = []
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            frames.append(frame if ok else None)

        per_cam = infer_rfdetr_with_margin(model, frames, args.conf, device)
        for cam_i, dets in enumerate(per_cam):
            if not dets:
                continue
            for d in dets:
                if d["class_id"] in args.focus:
                    name = names[d["class_id"]] if d["class_id"] < len(names) else f"cls_{d['class_id']}"
                    emit(f"{t:7.1f} {cam_i:3d} {name:<40} {d['confidence']:6.3f} {d['margin']:7.3f}")
        t += args.step_sec

    for cap in caps:
        cap.release()
    if out_f:
        out_f.close()
        print(f"\n저장됨: {args.out}")


if __name__ == "__main__":
    main()
