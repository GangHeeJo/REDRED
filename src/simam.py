"""
SimAM: Simple, Parameter-Free Attention Module (ICML 2021, Yang et al.)

파라미터 0개. 채널+공간 3D attention 동시 처리.
BP-YOLO (IEEE Access 2024)에서 YOLOv7에 적용해 mAP +8% 달성.

Usage (학습 스크립트 맨 위):
    import src.patch_ultralytics  # tasks.py globals에 SimAM 등록
    from ultralytics import YOLO
    model = YOLO("data/yolo11-simam.yaml")
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = h * w - 1  # unbiased estimate
        x_minus_mu = x - x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_sq = x_minus_mu.pow(2)
        y = x_minus_mu_sq / (
            4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return x * self.act(y)
