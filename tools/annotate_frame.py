"""
특정 카메라/시각의 프레임에 검출 bbox를 그려서 저장 -- 어느 물건이 어떤
클래스로 인식되는지 육안으로 확인하기 위함.

Usage:
    python tools/annotate_frame.py \
        --weights runs/rfdetr/checkpoint_best_total.pth \
        --cam 1 --sec 90 --conf 0.35 \
        --out output/annotated_cam1_t90.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from infer_rfdetr import load_rfdetr, infer_rfdetr


def load_names(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--names", default="data/names.txt")
    p.add_argument("--cam", type=int, required=True)
    p.add_argument("--sec", type=float, required=True)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    names = load_names(args.names)
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    video = str(Path.home() / f"Dataset/4.TestVideo_Sample/cam{args.cam}/Sample_1.mp4")

    print("모델 로딩...")
    model = load_rfdetr(args.weights, num_classes=len(names), device=device)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_no = int(args.sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("프레임 읽기 실패")
        return

    frames = [None] * 5
    frames[args.cam] = frame
    per_cam = infer_rfdetr(model, frames, args.conf, device)
    dets = per_cam[args.cam] or []

    print(f"검출 {len(dets)}개:")
    for d in sorted(dets, key=lambda x: -x["confidence"]):
        name = names[d["class_id"]] if d["class_id"] < len(names) else f"cls_{d['class_id']}"
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        conf = d["confidence"]
        print(f"  {name:<45} conf={conf:.3f} bbox=({x1},{y1},{x2},{y2})")
        color = (0, 0, 255) if conf >= 0.5 else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, frame)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
