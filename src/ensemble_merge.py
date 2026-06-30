"""
두 모델(YOLOv7 + YOLO11m)의 per-camera 검출 결과를 병합.

전략: 클래스별로 신뢰도 내림차순 정렬 후 NMS-style 중복 제거.
- 두 모델이 같은 객체를 감지하면 → 더 높은 confidence 박스 유지
- 한 모델만 감지한 객체는 → 그대로 포함 (recall 향상 핵심)

사용 목적: YOLO11m이 occlusion으로 놓치는 프레임을 YOLOv7이 보완.
"""

from collections import defaultdict
from typing import List, Optional, Dict

Detection = Dict          # {class_id, confidence, bbox:[x1,y1,x2,y2]}
PerCamDets = List[Optional[List[Detection]]]


def _iou(b1, b2) -> float:
    if not b1 or not b2:
        return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0


def ensemble_single_cam(
    dets1: List[Detection],
    dets2: List[Detection],
    iou_thr: float = 0.5,
) -> List[Detection]:
    """
    두 모델의 단일 카메라 검출 결과 병합.
    클래스별로 confidence 내림차순 정렬 후 IoU >= iou_thr인 박스는 중복 제거.
    """
    by_class: Dict[int, List[Detection]] = defaultdict(list)
    for d in dets1 + dets2:
        by_class[d["class_id"]].append(d)

    result = []
    for cls_dets in by_class.values():
        cls_dets.sort(key=lambda d: d["confidence"], reverse=True)
        kept: List[Detection] = []
        for det in cls_dets:
            bbox = det.get("bbox") or []
            if not bbox:
                kept.append(det)
                continue
            if not any(_iou(bbox, k["bbox"]) >= iou_thr for k in kept if k.get("bbox")):
                kept.append(det)
        result.extend(kept)

    return result


def ensemble_merge(
    per_cam1: PerCamDets,
    per_cam2: PerCamDets,
    iou_thr: float = 0.5,
) -> PerCamDets:
    """5개 카메라 전체에 대해 두 모델 결과를 병합."""
    merged = []
    for dets1, dets2 in zip(per_cam1, per_cam2):
        d1 = dets1 or []
        d2 = dets2 or []
        if not d1 and not d2:
            merged.append(None)
        else:
            merged.append(ensemble_single_cam(d1, d2, iou_thr))
    return merged
