"""
Multi-view fusion: combine detections from up to 5 cameras per frame.

Strategy: confidence-weighted voting per class.
  - Each camera votes on how many items of each class are visible.
  - Final count = weighted median of per-camera counts (robust to outlier cameras).

Alternative strategies are also provided for comparison.

Per-class override: weighted median requires a majority (3+/5) of cameras to
agree simultaneously, which structurally floors the count to 0 for items only
ever visible from a minority of camera angles -- regardless of how confidently
those cameras detect it. bumblebee_albacore/dove_white/dove_pink were confirmed
(2026-06-23, tools/probe_low_confidence.py) to never have 3+ cameras agree even
at production confidence (max 2/5), despite frequent high-confidence
single/double-camera detections. These classes use max-across-cameras instead.
"""

from typing import List, Dict, Optional, Set
import numpy as np
from collections import defaultdict


DetectionList = List[Dict]   # [{class_id, confidence, bbox}, ...]

# Classes only ever visible from a minority of cameras -- see module docstring.
MAX_CONFIDENCE_CLASS_IDS: Set[int] = {2, 53, 54}  # bumblebee_albacore, dove_pink, dove_white


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
    cam_weights: Optional[List[float]] = None,
    max_confidence_class_ids: Optional[Set[int]] = None,
) -> Dict[int, int]:
    """
    per_cam_detections: one DetectionList per camera (None if camera offline).
    cam_weights: importance of each camera (default: equal).
    max_confidence_class_ids: classes fused via max-across-cameras instead of
        weighted median (see module docstring). Defaults to MAX_CONFIDENCE_CLASS_IDS.
    Returns final integer count per class.
    """
    active = [(i, d) for i, d in enumerate(per_cam_detections) if d is not None]
    if not active:
        return {}

    if cam_weights is None:
        cam_weights = [1.0] * len(per_cam_detections)
    if max_confidence_class_ids is None:
        max_confidence_class_ids = MAX_CONFIDENCE_CLASS_IDS

    all_classes = set()
    for _, dets in active:
        all_classes.update(d["class_id"] for d in dets)

    result: Dict[int, int] = {}
    for cls_id in all_classes:
        votes = []
        weights = []
        for cam_idx, dets in active:
            cnt = sum(1 for d in dets if d["class_id"] == cls_id)
            votes.append(cnt)
            weights.append(cam_weights[cam_idx])

        if cls_id in max_confidence_class_ids:
            result[cls_id] = max(votes)
            continue

        # Weighted median
        votes = np.array(votes, dtype=float)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        sorted_idx = np.argsort(votes)
        cumsum = np.cumsum(weights[sorted_idx])
        median_val = votes[sorted_idx[np.searchsorted(cumsum, 0.5)]]
        result[cls_id] = int(round(median_val))

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
        # Most common count
        result[cls_id] = max(set(counts), key=counts.count)
    return result


# Default fusion function used by the pipeline
fuse = fuse_weighted_median
