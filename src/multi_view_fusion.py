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
# 2026-07-02: RF-DETR 진단 중 전부 비웠다가(과다발화 milano는 고쳐졌지만
# crystal_hot_sauce가 새로 미탐지로 회귀함) 원인 분리함 -- 이 표엔 성격이 다른
# 두 종류가 섞여 있었음:
#   (a) "카메라 1~2대에서만 물리적으로 보이는" 구조적 케이스(bumblebee/dove_pink/
#       redbull/crystal_hot_sauce/dr_pepper/spam) — 카메라 배치 문제라 모델과
#       무관하게 여전히 필요. 복원함.
#   (b) "YOLOv7이 잘 못 잡아서 문턱을 낮춘" 모델별 케이스(milano/dove_white) —
#       RF-DETR은 반대로 노이즈 섞어 잡아서 quorum=1이 과다발화 유발. 계속 비워두고
#       RF-DETR 전용 값은 tools/analyze_per_cam.py(output/per_cam_rfdetr.csv)로
#       따로 산출할 것. milano는 cam-weight exclusion만으로 이미 해결됨
#       (run_pipeline_rfdetr.py 참고). dove_white는 quorum=2+whitelist없음 조합이
#       완전 미탐지로 회귀했으니 재산출 필요.
CLASS_QUORUM_OVERRIDE: Dict[int, int] = {
    2:  2,   # bumblebee_albacore
    53: 1,   # dove_pink
    15: 1,   # redbull
    39: 1,   # crystal_hot_sauce
    21: 1,   # dr_pepper
    29: 2,   # spam
    # 54: dove_white — RF-DETR 전용 값 재산출 전까지 비워둠
    # 42: pepperidge_farm_milano — cam-weight exclusion으로 해결됨, quorum 불필요
}

# RF-DETR 전용 값 (2026-07-02, output/per_cam_rfdetr.csv + tools/analyze_per_cam.py
# --focus로 산출, 검출률 >=5% 카메라만 채택).
# milano(42)는 cam-weight exclusion만으로 이미 해결돼서 추가 안 함(회귀 방지).
# campbells(43)/chewy_dips_peanut_butter(46)는 스크립트가 "whitelist 불필요"로
# 판정 — campbells는 전카메라 4.3% 이하 구조적 미탐지(YOLOv7과 동일 결론),
# chewy_dips는 5대 골고루 보여서 카메라 문제가 아니라 순수 confidence flicker
# (quorum/whitelist로 해결 안 됨, per_class_confirm 대상).
CLASS_CAM_WHITELIST: Dict[int, List[int]] = {
    0:  [0, 2],     # aunt_jemima_original_syrup
    4:  [0, 1, 3],  # crayola_24_crayons
    17: [0, 3, 4],  # a1_steak_sauce
    31: [0, 2],     # pepperidge_farm_milk_chocolate_macadamia_cookies
    36: [1, 2, 3],  # nature_valley_crunchy_oats_n_honey
    41: [2],        # nabisco_nilla_wafers
    54: [3],        # dove_white
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
