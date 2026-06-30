"""
Multi-view fusion: combine detections from up to 5 cameras per frame.

Strategy:
  1. Per-class occlusion detection: cameras that report 0 confidence for a
     class while >=min_corroborate others report >0 are excluded (weight=0).
  2. Global quorum vote: among remaining cameras, take the quorum-th highest
     per-camera count. quorum=1 means any single camera suffices; quorum=2
     means at least 2 must agree; quorum=3 ≈ old weighted-median behaviour.

Tuning knobs (no class-specific overrides):
  quorum          — cameras that must agree (default 2)
  min_corroborate — others needed to confirm before excluding a camera (default 2)
"""

from typing import List, Dict, Optional, Union
import numpy as np
from collections import defaultdict


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
    quorum: take the quorum-th highest vote among active cameras.
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
        cls_weights = cam_weights.get(cls_id, default_weights) if per_class_weights else cam_weights
        votes = [
            sum(1 for d in dets if d["class_id"] == cls_id)
            for cam_idx, dets in active
            if cls_weights[cam_idx] != 0.0
        ]
        if not votes:
            continue
        sorted_desc = sorted(votes, reverse=True)
        idx = min(quorum, len(sorted_desc)) - 1
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
