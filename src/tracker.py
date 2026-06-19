"""
SORT (Simple Online and Realtime Tracking) 기반 다중 카메라 객체 추적기.

각 카메라별로 독립적으로 추적하며, 확정된(confirmed) track만
EventDetector로 전달하여 오탐 감소.

기존 방식과 비교:
    카운팅 방식: 프레임마다 개수 세고 변화 감지 (노이즈에 취약)
    트래킹 방식: 개별 객체에 ID를 부여하고 추적
                 min_hits 이상 연속 감지된 track만 확정
                 → 안정적인 재고 수량 제공

SORT 알고리즘:
    1. Kalman filter로 각 track 위치 예측
    2. Hungarian algorithm으로 탐지 결과와 매칭 (같은 클래스끼리만)
    3. min_hits 이상 연속 감지된 track만 확정 (confirmed)
    4. max_age 프레임 이상 미감지 track 삭제

Usage:
    tracker = MultiCameraTracker(n_cameras=5)
    confirmed_dets = tracker.update(per_cam_detections)
    # confirmed_dets는 기존 per_cam_detections와 동일한 포맷
    # → fuse(), EventDetector 코드 변경 없이 그대로 사용 가능
"""

import numpy as np
from typing import Dict, List, Optional

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------
# IoU 및 매칭
# ---------------------------------------------------------------

def _iou(b1, b2):
    """IoU between two bboxes [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter + 1e-6)


def _match(iou_matrix, threshold):
    """Hungarian or greedy matching. Returns [(det_idx, trk_idx), ...]."""
    if iou_matrix.size == 0:
        return []
    if _SCIPY:
        rows, cols = linear_sum_assignment(1 - iou_matrix)
        return [(r, c) for r, c in zip(rows, cols)
                if iou_matrix[r, c] >= threshold]
    # greedy fallback
    pairs, used_r, used_c = [], set(), set()
    flat = sorted(
        [(iou_matrix[r, c], r, c)
         for r in range(iou_matrix.shape[0])
         for c in range(iou_matrix.shape[1])],
        reverse=True,
    )
    for val, r, c in flat:
        if val < threshold:
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((r, c)); used_r.add(r); used_c.add(c)
    return pairs


# ---------------------------------------------------------------
# Kalman 기반 단일 객체 Tracker
# ---------------------------------------------------------------

class KalmanBoxTracker:
    """
    단일 bbox를 constant-velocity Kalman filter로 추적.
    state: [cx, cy, w, h, dcx, dcy, dw, dh]
    """
    _counter = 0

    def __init__(self, bbox, class_id: int):
        self.id       = KalmanBoxTracker._counter
        KalmanBoxTracker._counter += 1
        self.class_id = class_id

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w  =  bbox[2] - bbox[0]
        h  =  bbox[3] - bbox[1]

        self._s = np.array([cx, cy, w, h, 0., 0., 0., 0.])
        self.hits              = 1
        self.age               = 0
        self.time_since_update = 0

    def predict(self):
        """Constant velocity 예측."""
        self._s[:4] += self._s[4:]
        self.age += 1
        self.time_since_update += 1
        return self._get_bbox()

    def update(self, bbox):
        """매칭된 탐지 결과로 상태 업데이트."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w  =  bbox[2] - bbox[0]
        h  =  bbox[3] - bbox[1]
        meas = np.array([cx, cy, w, h])
        alpha = 0.6
        self._s[4:] = alpha * self._s[4:] + (1 - alpha) * (meas - self._s[:4])
        self._s[:4] = meas
        self.hits += 1
        self.time_since_update = 0

    def _get_bbox(self):
        cx, cy, w, h = self._s[:4]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def to_detection(self):
        """확정된 track을 YOLO detection dict 포맷으로 반환."""
        return {
            "class_id":   self.class_id,
            "confidence": 1.0,
            "bbox":       self._get_bbox(),
            "track_id":   self.id,
        }


# ---------------------------------------------------------------
# 단일 카메라 SORT Tracker
# ---------------------------------------------------------------

class Sort:
    """
    단일 카메라용 SORT 트래커.

    Args:
        max_age      : 미감지 허용 최대 프레임 수 (초과 시 track 삭제)
        min_hits     : 확정(confirmed)까지 필요한 연속 감지 횟수
        iou_threshold: 탐지-track 매칭 최소 IoU
    """

    def __init__(self, max_age: int = 3, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: [{"class_id", "confidence", "bbox": [x1,y1,x2,y2]}, ...]
        Returns   : 확정된 track들을 detection dict 포맷으로 반환
                    (fuse(), EventDetector와 동일한 포맷 → 기존 코드 변경 없음)
        """
        # 1. 기존 track 예측
        predicted = [t.predict() for t in self.trackers]

        # 2. 탐지 - track 매칭 (같은 클래스끼리만)
        matched_det, matched_trk = set(), set()

        if self.trackers and detections:
            n_d, n_t = len(detections), len(self.trackers)
            iou_mat  = np.zeros((n_d, n_t))
            for d_i, det in enumerate(detections):
                for t_i, trk in enumerate(self.trackers):
                    if det["class_id"] == trk.class_id:
                        iou_mat[d_i, t_i] = _iou(det["bbox"], predicted[t_i])

            for d_i, t_i in _match(iou_mat, self.iou_threshold):
                self.trackers[t_i].update(detections[d_i]["bbox"])
                matched_det.add(d_i)
                matched_trk.add(t_i)

        # 3. 매칭 안 된 탐지 → 새 track 생성
        for d_i, det in enumerate(detections):
            if d_i not in matched_det:
                self.trackers.append(
                    KalmanBoxTracker(det["bbox"], det["class_id"])
                )

        # 4. 오래된 track 삭제
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # 5. 확정된 track만 반환
        return [
            t.to_detection()
            for t in self.trackers
            if t.hits >= self.min_hits and t.time_since_update == 0
        ]


# ---------------------------------------------------------------
# 다중 카메라 Tracker
# ---------------------------------------------------------------

class MultiCameraTracker:
    """
    카메라별로 독립적인 Sort 트래커 적용.
    run_pipeline.py에서 --use_tracker 플래그로 활성화.

    Args:
        n_cameras    : 카메라 수 (기본 5)
        max_age      : Sort.max_age
        min_hits     : Sort.min_hits (높을수록 보수적)
        iou_threshold: Sort.iou_threshold
    """

    def __init__(self, n_cameras: int = 5, max_age: int = 3,
                 min_hits: int = 3, iou_threshold: float = 0.3):
        self.trackers = [
            Sort(max_age=max_age, min_hits=min_hits,
                 iou_threshold=iou_threshold)
            for _ in range(n_cameras)
        ]

    def update(self, per_cam_detections: List[Optional[List[Dict]]]) \
            -> List[Optional[List[Dict]]]:
        """
        per_cam_detections: [cam0_dets, cam1_dets, ...] (None = 카메라 오프라인)
        Returns: 각 카메라별 confirmed track detection list
                 포맷이 동일하므로 fuse()에 그대로 전달 가능
        """
        result = []
        for cam_idx, dets in enumerate(per_cam_detections):
            if dets is None:
                result.append(None)
            else:
                result.append(self.trackers[cam_idx].update(dets))
        return result
