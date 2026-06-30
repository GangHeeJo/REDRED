"""
YOLO11 + SimAM fine-tuning 스크립트.

SimAM은 파라미터 0개 → 기존 학습된 YOLO11 가중치에 hook으로 얹고 fine-tuning.
처음부터 재학습 불필요.

서버 실행:
    cd ~/REDRED
    # 기존 YOLOv7 학습 결과를 YOLO11로 변환한 best.pt 필요
    nohup python tools/train_simam.py --weights runs/train/exp/weights/best.pt > train_simam.log 2>&1 &
    tail -f train_simam.log

    # 또는 ultralytics 사전학습 가중치로 시작:
    nohup python tools/train_simam.py --weights yolo11n.pt > train_simam.log 2>&1 &
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from src.patch_ultralytics import apply_simam_hooks

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="yolo11n.pt",
                    help="시작 가중치 (ultralytics 공식 pt 또는 기존 학습된 best.pt)")
parser.add_argument("--epochs",  type=int, default=50)
parser.add_argument("--batch",   type=int, default=16)
parser.add_argument("--device",  default="0")
args = parser.parse_args()

model = YOLO(args.weights)
apply_simam_hooks(model)
print("SimAM hooks applied to P3/P4/P5 (layers 16, 19, 22)")

model.train(
    data="data/custom.yaml",
    epochs=args.epochs,
    imgsz=640,
    batch=args.batch,
    device=args.device,
    workers=8,
    project="runs/simam",
    name="exp",
    exist_ok=False,
)
