"""
ultralytics tasks.py의 globals()에 SimAM을 등록하는 패치.

학습 스크립트 또는 run_pipeline.py 맨 위에서:
    import src.patch_ultralytics
이 한 줄로 yolo11-simam.yaml을 YOLO()에서 정상 파싱할 수 있음.
"""

import ultralytics.nn.tasks as _tasks
from src.simam import SimAM  # noqa: F401

_tasks.SimAM = SimAM
