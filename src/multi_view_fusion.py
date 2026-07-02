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
#
# 2026-07-02: RF-DETR 진단 위해 임시로 비움. 아래 값들은 YOLOv7(main)용으로
# 튜닝된 값이라 RF-DETR 감지 패턴과 안 맞을 수 있음 (예: 42/54는 YOLOv7이
# 거의 못 잡아서 quorum=1로 낮춘 건데, RF-DETR은 반대로 이 클래스들을
# 노이즈 섞어서 잡아서 quorum=1이 과다발화를 유발하는 것으로 의심됨).
# RF-DETR 전용 값은 tools/analyze_per_cam.py로 다시 산출할 것.
#
# CLASS_QUORUM_OVERRIDE: Dict[int, int] = {
#     2:  2,   # bumblebee_albacore
#     53: 1,   # dove_pink
#     54: 1,   # dove_white
#     15: 1,   # redbull
#     39: 1,   # crystal_hot_sauce
#     21: 1,   # dr_pepper
#     29: 2,   # spam
#     42: 1,   # pepperidge_farm_milano
# }
CLASS_QUORUM_OVERRIDE: Dict[int, int] = {}

# CLASS_CAM_WHITELIST: Dict[int, List[int]] = {
#     43: [0],     # campbells_chicken_noodle_soup: cam4 chunky 혼동 차단
#     42: [3, 4],  # pepperidge_farm_milano: 노이즈 cam0 제거
#     54: [3],     # dove_white: cam3만
# }
CLASS_CAM_WHITELIST: Dict[int, List[int]] = {}


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
