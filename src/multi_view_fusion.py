"""
Multi-view fusion: combine detections from up to 5 cameras per frame.

Strategy:
  1. Per-class occlusion detection: cameras that report 0 confidence for a
     class while >=min_corroborate others report >0 are excluded (weight=0).
  2. Quorum vote: among remaining cameras, take the quorum-th highest
     per-camera count. CLASS_QUORUM_OVERRIDE allows per-class override.
  3. CLASS_CAM_WHITELIST: restrict which cameras vote for specific classes.

Tuning knobs:
  quorum              — global default (default 2)
  min_corroborate     — occlusion exclusion threshold (default 2)
  CLASS_QUORUM_OVERRIDE — {class_id: quorum} overrides global quorum per class
  CLASS_CAM_WHITELIST   — {class_id: [cam_ids]} restricts voting cameras

Camera layout:  0=왼앞  1=오른앞  2=위(top)  3=오른뒤  4=왼뒤
"""

from typing import List, Dict, Optional, Union
import numpy as np
from collections import defaultdict


# ── 클래스별 quorum 오버라이드 ────────────────────────────────────────
# 기본값(global quorum=2)에서 벗어나야 하는 클래스만 등록
#
# quorum=1: 1대 카메라만 감지해도 인정 (1~2대에서만 보이는 상품)
# quorum=3: 3대 이상 동의해야 인정 (중복 발화되는 상품)
CLASS_QUORUM_OVERRIDE: Dict[int, int] = {
    # ── 1대 전용 카메라 (whitelist 1대 → quorum=1 고정) ──
    5:  1,   # hersheys_cocoa       → whitelist=[1]
    14: 1,   # hersheys_bar         → whitelist=[3]
    15: 1,   # redbull              → whitelist=[0]
    21: 1,   # dr_pepper            → whitelist=[4]
    23: 1,   # bulls_eye_bbq        → whitelist=[3]
    38: 1,   # palmolive_orange     → whitelist=[3]
    39: 1,   # crystal_hot_sauce    → whitelist=[3]
    # ── 2대 whitelist → quorum=2 (두 카메라 모두 동의) ──
    3:  2,   # cholula              → whitelist=[3,4]
    8:  2,   # hunts_sauce          → whitelist=[0,3]
    # ── 5대 모두 보임 → quorum=3 유지 ──
    28: 3,   # quaker_big_chewy_chocolate_chip
    # campbells_chicken_noodle_soup(43): 모델이 전혀 검출 안 함 → 파이프라인 불가
}

# ── 클래스별 카메라 화이트리스트 (per_cam_log 분석 기반) ─────────────
# cam layout: 0=왼앞  1=오른앞  2=위(top)  3=오른뒤  4=왼뒤
CLASS_CAM_WHITELIST: Dict[int, List[int]] = {
    3:  [3, 4],  # cholula       cam3(9%), cam4(8.4%)
    5:  [1],     # hersheys_cocoa  cam1(3%)만 검출
    8:  [0, 3],  # hunts_sauce   cam0(30%), cam3(26%)
    14: [3],     # hersheys_bar  cam3(6.4%)만
    15: [0],     # redbull       cam0(7.1%)만
    21: [4],     # dr_pepper     cam4(9.9%) 주도
    23: [3],     # bulls_eye_bbq cam3(1.1%)만
    38: [3],     # palmolive     cam3(0.5%)만
    39: [3],     # crystal_hot   cam3(3.6%)만
}


DetectionList = List[Dict]   # [{class_id, confidence, bbox}, ...]


def count_per_class(detections: DetectionList) -> Dict[int, float]:
    """Sum confidence scores per class as a soft count."""
    scores: Dict[int, float] = defaultdict(float)
    for det in detections:
        scores[det["class_id"]] += det["confidence"]
    return scores


def hard_count_per_class(detections: DetectionList) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for det in detections:
        counts[det["class_id"]] += 1
    return counts


def fuse_weighted_median(
    per_cam_detections: List[Optional[DetectionList]],
    cam_weights: Optional[Union[List[float], Dict[int, List[float]]]] = None,
    quorum: int = 2,
) -> Dict[int, int]:
    """
    per_cam_detections: one DetectionList per camera (None if camera offline).
    cam_weights: cameras with weight=0 are excluded from voting. Either a flat
        list (same for all classes) or {class_id: [w0,...]} from
        compute_per_class_cam_weights() for automatic per-class occlusion detection.
    quorum: global default. CLASS_QUORUM_OVERRIDE takes precedence per class.
    CLASS_CAM_WHITELIST: non-whitelisted cameras get weight=0 for that class.
    Returns final integer count per class.
    """
    active = [(i, d) for i, d in enumerate(per_cam_detections) if d is not None]
    if not active:
        return {}

    n = len(per_cam_detections)
    default_weights = [1.0] * n
    per_class_weights = isinstance(cam_weights, dict)
    if cam_weights is None:
        cam_weights = default_weights

    all_classes: set = set()
    for _, dets in active:
        all_classes.update(d["class_id"] for d in dets)

    result: Dict[int, int] = {}
    for cls_id in all_classes:
        cls_quorum = CLASS_QUORUM_OVERRIDE.get(cls_id, quorum)
        cls_weights = list(
            cam_weights.get(cls_id, default_weights) if per_class_weights else cam_weights
        )

        # Apply cam whitelist: zero out cameras not in the whitelist
        if cls_id in CLASS_CAM_WHITELIST:
            whitelist = CLASS_CAM_WHITELIST[cls_id]
            cls_weights = [w if cam_idx in whitelist else 0.0
                           for cam_idx, w in enumerate(cls_weights)]

        votes = [
            sum(1 for d in dets if d["class_id"] == cls_id)
            for cam_idx, dets in active
            if cls_weights[cam_idx] != 0.0
        ]
        if not votes:
            continue
        sorted_desc = sorted(votes, reverse=True)
        idx = min(cls_quorum, len(sorted_desc)) - 1
        result[cls_id] = sorted_desc[idx]

    return result


def fuse_max_confidence(
    per_cam_detections: List[Optional[DetectionList]],
) -> Dict[int, int]:
    """Take the maximum count across cameras — optimistic, risks overcounting."""
    result: Dict[int, int] = defaultdict(int)
    for dets in per_cam_detections:
        if dets is None:
            continue
        for cls_id, cnt in hard_count_per_class(dets).items():
            result[cls_id] = max(result[cls_id], cnt)
    return dict(result)


def fuse_majority_vote(
    per_cam_detections: List[Optional[DetectionList]],
) -> Dict[int, int]:
    """Simple majority: count how many cameras agree on each class count."""
    active = [d for d in per_cam_detections if d is not None]
    if not active:
        return {}

    all_classes: set = set()
    for dets in active:
        all_classes.update(d["class_id"] for d in dets)

    result: Dict[int, int] = {}
    for cls_id in all_classes:
        counts = [sum(1 for d in dets if d["class_id"] == cls_id) for dets in active]
        result[cls_id] = max(set(counts), key=counts.count)
    return result


# Default fusion function used by the pipeline
fuse = fuse_weighted_median
