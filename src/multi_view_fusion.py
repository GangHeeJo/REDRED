"""
Multi-view fusion: combine detections from up to 5 cameras per frame.

Strategy: confidence-weighted voting per class.
  - Each camera votes on how many items of each class are visible.
  - Final count = weighted median of per-camera counts (robust to outlier cameras).

Alternative strategies are also provided for comparison.

Per-class quorum override: weighted median requires a majority (3+/5) of
cameras to agree simultaneously, which structurally floors the count to 0 for
items only ever visible from a minority of camera angles -- regardless of how
confidently those cameras detect it. bumblebee_albacore/dove_white/dove_pink
were confirmed (2026-06-23, tools/probe_low_confidence.py) to never have 3+
cameras agree even at production confidence (max 2/5), despite frequent
high-confidence single/double-camera detections. These classes use a lower
quorum (the N-th highest per-camera vote instead of the full median).

bumblebee_albacore/dove_pink: quorum=1 (any single camera, i.e. max-across-
cameras) -- clean in practice, no extra false positives observed.
dove_white: quorum=2 (2026-06-23) -- quorum=1 alone caused 4 spurious
duplicate events from single-camera noise blips (glare/reflection on the
white soap reading as a brief false positive); dove_white does have genuine
2-camera agreement (37% of frames when present), so requiring 2 keeps real
events while filtering single-camera flicker.

redbull/crystal_hot_sauce/dr_pepper: quorum=1 (2026-06-23, same probe tool,
re-run after these 3 + campbells_chicken_noodle_soup showed up FN in
ground_truth_v2 scoring). All three are detected continuously and cleanly
from frame 0 up to right before their GT purchase time, but only by 1-2
specific cameras the whole time (redbull: cam0 only; crystal_hot_sauce:
cam3, occasionally +cam4; dr_pepper: cam4, joined by cam1 then briefly cam0
near t=21-22s) -- never the 3/5 majority weighted-median needs, so the
fused count was floored to 0 for the entire video and the purchase event
(1->0) never had a baseline to drop from. No flicker observed in the
single-camera signal for any of the three, so quorum=1 is not expected to
introduce noise the way it did for dove_white.

spam: quorum=2 (2026-06-24, probe3). Reaches max 3 simultaneous cameras at
conf=0.05 with many 1-2 camera frames; quorum=2 recovers those windows and
confirmed clean (no spurious events) in the follow-up run.

pepperidge_farm_milano_cookies_double_chocolate: also reaches max 3 cameras,
but quorum=2 caused 5x duplicate purchase+return events (fused count
oscillating 1<->0 across 2-camera agreements). Reverted to default median.
Needs a different fix (e.g. per-class CONFIRM_FRAMES or signal smoothing).

haribo_gold_bears_gummi_candy/bulls_eye_bbq_sauce_original: NOT a quorum
problem. haribo reaches 5/5 cameras simultaneously (mean_conf=0.739 at
conf=0.05), bulls_eye reaches 5/5 cameras (max_conf=0.842). Quorum override
would not fix them; their failures are due to event-detection timing or GPU
non-determinism. Deliberately excluded.

campbells_chicken_noodle_soup was probed at the same time and shows the
identical low-camera-count signature pre-purchase, but cam4 keeps reporting
it continuously for ~100s *after* its GT purchase time (11s) -- almost
certainly confusion with the visually similar campbells_chunky_classic_-
chicken_noodle rather than a quorum problem. Deliberately left out of the
override here; needs a bbox-position check before touching its fusion.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict


DetectionList = List[Dict]   # [{class_id, confidence, bbox}, ...]

# class_id -> minimum number of cameras that must agree for a class to be
# fused via "N-th highest vote" instead of the full weighted median. See
# module docstring for why each of these needs a lower quorum than the
# camera-majority default.
CLASS_QUORUM_OVERRIDE: Dict[int, int] = {
    2:  1,   # bumblebee_albacore
    53: 1,   # dove_pink
    54: 2,   # dove_white
    15: 1,   # redbull
    39: 1,   # crystal_hot_sauce
    21: 1,   # dr_pepper
    29: 2,   # spam
}


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
    class_quorum_override: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """
    per_cam_detections: one DetectionList per camera (None if camera offline).
    cam_weights: importance of each camera (default: equal).
    class_quorum_override: class_id -> minimum number of agreeing cameras,
        fused via "quorum-th highest vote" instead of weighted median (see
        module docstring). Defaults to CLASS_QUORUM_OVERRIDE.
    Returns final integer count per class.
    """
    active = [(i, d) for i, d in enumerate(per_cam_detections) if d is not None]
    if not active:
        return {}

    if cam_weights is None:
        cam_weights = [1.0] * len(per_cam_detections)
    if class_quorum_override is None:
        class_quorum_override = CLASS_QUORUM_OVERRIDE

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

        if cls_id in class_quorum_override:
            quorum = class_quorum_override[cls_id]
            sorted_desc = sorted(votes, reverse=True)
            idx = min(quorum, len(sorted_desc)) - 1
            result[cls_id] = sorted_desc[idx]
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
