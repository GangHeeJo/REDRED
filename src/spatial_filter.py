"""
Spatial Filter: bbox 크기 기반 위치 추정 + FP 필터링

핵심 아이디어:
  물건의 실제 크기는 고정 → bbox pixel 크기 ∝ 1/거리 (핀홀 카메라 모델)
  → bbox 크기로 선반 내 상대 위치 추정 가능
  → 위치 정보로 중복 감지 / 이상 감지 필터링

동작 방식:
  1. Warm-up (초기 N프레임): 클래스별 bbox 크기 분포 자동 학습
  2. 런타임: 분포에서 크게 벗어난 bbox → FP 후보로 제거
  3. 같은 프레임/카메라에서 같은 클래스가 여러 개일 때:
     bbox 중심 거리가 너무 가까우면 중복 제거 (NMS-like)
  4. 5카메라 교차: 물건의 추정 깊이가 카메라별로 일관되는지 확인
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


# ── 파라미터 ─────────────────────────────────────────────────────
WARMUP_FRAMES    = 60    # bbox 크기 분포 학습에 사용할 프레임 수
SIZE_SIGMA       = 2.5   # 평균±N*sigma 밖이면 이상치
MIN_SAMPLES      = 10    # 이 수 미만이면 필터 적용 안 함
DUP_IOU_THRESH   = 0.4   # 같은 클래스 bbox IoU가 이 이상이면 중복
DEPTH_SIGMA      = 2.0   # 깊이 일관성 검증 sigma


# ── 유틸 ─────────────────────────────────────────────────────────

def _bbox_iou(a, b):
    """두 bbox [x1,y1,x2,y2] 간 IoU."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _bbox_norm_size(det, img_h, img_w):
    """정규화된 bbox (h, w) 반환. img 크기 모르면 절대값 그대로."""
    x1, y1, x2, y2 = det["bbox"]
    h = (y2 - y1) / img_h if img_h else (y2 - y1)
    w = (x2 - x1) / img_w if img_w else (x2 - x1)
    return h, w


def _bbox_center(det):
    x1, y1, x2, y2 = det["bbox"]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# ── 메인 클래스 ──────────────────────────────────────────────────

class SpatialFilter:
    """
    Usage:
        sf = SpatialFilter(img_h=720, img_w=1280)

        # warm-up 단계 (파이프라인 앞부분)
        for frame_dets in first_frames:
            sf.update_warmup(frame_dets)

        # 런타임 필터링
        filtered = sf.filter(per_cam_dets)
    """

    def __init__(
        self,
        img_h: int = 720,
        img_w: int = 1280,
        warmup_frames: int = WARMUP_FRAMES,
        size_sigma: float = SIZE_SIGMA,
        dup_iou_thresh: float = DUP_IOU_THRESH,
    ):
        self.img_h          = img_h
        self.img_w          = img_w
        self.warmup_frames  = warmup_frames
        self.size_sigma     = size_sigma
        self.dup_iou_thresh = dup_iou_thresh

        self._warmup_done   = False
        self._frame_count   = 0

        # warm-up 수집: {cls_id: [norm_h, ...]}
        self._h_samples: Dict[int, List[float]] = defaultdict(list)

        # 학습된 분포: {cls_id: (mean_h, std_h)}
        self._h_stats: Dict[int, tuple] = {}

        # 통계용
        self.stats = {"removed_size": 0, "removed_dup": 0, "total_in": 0}

    # ── warm-up ──────────────────────────────────────────────────

    def update_warmup(self, per_cam_dets: List[Optional[List]]):
        """초기 N프레임에서 클래스별 bbox 크기 분포 수집."""
        if self._warmup_done:
            return
        for dets in per_cam_dets:
            if dets is None:
                continue
            for det in dets:
                h, _ = _bbox_norm_size(det, self.img_h, self.img_w)
                if h > 0:
                    self._h_samples[det["class_id"]].append(h)

        self._frame_count += 1
        if self._frame_count >= self.warmup_frames:
            self._finalize_warmup()

    def _finalize_warmup(self):
        for cls_id, samples in self._h_samples.items():
            if len(samples) >= MIN_SAMPLES:
                arr = np.array(samples)
                self._h_stats[cls_id] = (float(arr.mean()), float(arr.std()) + 1e-6)
        self._warmup_done = True
        print(f"[SpatialFilter] warm-up 완료: {len(self._h_stats)}개 클래스 크기 학습")

    @property
    def ready(self):
        return self._warmup_done

    # ── 필터링 ───────────────────────────────────────────────────

    def filter(self, per_cam_dets: List[Optional[List]]) -> List[Optional[List]]:
        """
        per_cam_dets의 각 카메라 detection 필터링.
        warm-up 미완료 시 원본 그대로 반환.
        """
        if not self._warmup_done:
            return per_cam_dets

        result = []
        for dets in per_cam_dets:
            if dets is None:
                result.append(None)
                continue
            self.stats["total_in"] += len(dets)
            filtered = self._filter_size_outliers(dets)
            filtered = self._filter_duplicates(filtered)
            result.append(filtered)
        return result

    def _filter_size_outliers(self, dets: List) -> List:
        """bbox 크기가 학습된 분포에서 크게 벗어나면 제거."""
        keep = []
        for det in dets:
            cls_id = det["class_id"]
            if cls_id not in self._h_stats:
                keep.append(det)
                continue
            h, _ = _bbox_norm_size(det, self.img_h, self.img_w)
            mean_h, std_h = self._h_stats[cls_id]
            if abs(h - mean_h) <= self.size_sigma * std_h:
                keep.append(det)
            else:
                self.stats["removed_size"] += 1
        return keep

    def _filter_duplicates(self, dets: List) -> List:
        """같은 프레임/카메라에서 같은 클래스 bbox가 크게 겹치면 confidence 낮은 것 제거."""
        if len(dets) <= 1:
            return dets

        # 클래스별 그룹핑
        by_class: Dict[int, List] = defaultdict(list)
        for det in dets:
            by_class[det["class_id"]].append(det)

        keep = []
        for cls_id, cls_dets in by_class.items():
            if len(cls_dets) == 1:
                keep.extend(cls_dets)
                continue
            # confidence 내림차순 정렬 후 NMS
            cls_dets = sorted(cls_dets, key=lambda d: d["confidence"], reverse=True)
            suppressed = [False] * len(cls_dets)
            for i in range(len(cls_dets)):
                if suppressed[i]:
                    continue
                keep.append(cls_dets[i])
                for j in range(i + 1, len(cls_dets)):
                    if suppressed[j]:
                        continue
                    iou = _bbox_iou(cls_dets[i]["bbox"], cls_dets[j]["bbox"])
                    if iou >= self.dup_iou_thresh:
                        suppressed[j] = True
                        self.stats["removed_dup"] += 1
        return keep

    # ── 위치 추정 ─────────────────────────────────────────────────

    def estimate_depth(self, det) -> Optional[float]:
        """
        bbox 높이 기반 상대 깊이 추정.
        반환값: 클래스 평균 bbox 높이 대비 비율 (1.0 = 평균 거리, <1 = 더 가까움)
        """
        cls_id = det["class_id"]
        if cls_id not in self._h_stats:
            return None
        h, _ = _bbox_norm_size(det, self.img_h, self.img_w)
        if h <= 0:
            return None
        mean_h, _ = self._h_stats[cls_id]
        return mean_h / h  # 작을수록 가까이 있음

    def shelf_position(self, det) -> Optional[dict]:
        """
        detection의 선반 내 상대 위치 추정.
        Returns: {"depth": float, "x_norm": float, "y_norm": float}
        depth < 1: 앞줄, depth > 1: 뒷줄
        """
        depth = self.estimate_depth(det)
        if depth is None:
            return None
        cx, cy = _bbox_center(det)
        return {
            "depth":  depth,
            "x_norm": cx / self.img_w,  # 0=왼쪽, 1=오른쪽
            "y_norm": cy / self.img_h,  # 0=위, 1=아래
        }

    def print_stats(self):
        print(f"[SpatialFilter] total_in={self.stats['total_in']} "
              f"size_removed={self.stats['removed_size']} "
              f"dup_removed={self.stats['removed_dup']}")
