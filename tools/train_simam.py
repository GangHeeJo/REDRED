"""
YOLO11 + SimAM 학습 스크립트.

서버 실행:
    cd ~/REDRED
    nohup python tools/train_simam.py > train_simam.log 2>&1 &
    tail -f train_simam.log
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.patch_ultralytics  # noqa: F401, E402 — SimAM을 ultralytics에 등록

from ultralytics import YOLO  # noqa: E402

model = YOLO("data/yolo11-simam.yaml")

model.train(
    data="data/custom.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    project="runs/simam",
    name="exp",
    exist_ok=False,
    pretrained=False,
)
