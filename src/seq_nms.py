"""
Online (causal) SeqNMS for video object detection.

Paper: Han et al., "Seq-NMS for Video Object Detection" (2016)
       https://arxiv.org/abs/1602.08465

Original offline SeqNMS links all bboxes globally in time.
This is a causal (real-time) approximation: for each frame, look back
`seq_len` frames and re-score confidence using the matched chain.

Isolated single-frame detections (chain_len < min_seq) get penalized,
effectively suppressing FP blips before they reach the tracker/fusion.

Usage in pipeline (per camera):
    nms = OnlineSeqNMS(seq_len=5, iou_thresh=0.4, min_seq=2, penalty=0.0)
    filtered = nms.update(raw_dets_from_infer_batch)
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional


DetectionList = List[Dict]


def _iou(b1: list, b2: list) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class OnlineSeqNMS:
    """
    Causal per-camera SeqNMS filter.

    Parameters
    ----------
    seq_len   : rolling look-back window (number of processed frames)
    iou_thresh: IoU required to link bbox across frames (same class)
    method    : 'max' — chain confidence = max over matched frames
                'avg' — chain confidence = mean over matched frames
    min_seq   : minimum chain length to keep at full confidence
    penalty   : confidence multiplier for chain_len < min_seq
                0.0 = fully suppress single-frame blips (default)
                0.5 = halve confidence
    """

    def __init__(
        self,
        seq_len: int = 5,
        iou_thresh: float = 0.4,
        method: str = "max",
        min_seq: int = 2,
        penalty: float = 0.0,
    ) -> None:
        self.seq_len = seq_len
        self.iou_thresh = iou_thresh
        self.method = method
        self.min_seq = min_seq
        self.penalty = penalty
        self._history: deque[DetectionList] = deque(maxlen=seq_len)

    def reset(self) -> None:
        self._history.clear()

    def update(self, detections: Optional[DetectionList]) -> Optional[DetectionList]:
        """
        Filter one frame's detections with causal SeqNMS.

        Parameters
        ----------
        detections: list of {class_id, confidence, bbox [x1,y1,x2,y2]},
                    or None if camera is offline.

        Returns
        -------
        Filtered detection list (same format, confidence re-scored).
        Detections with re-scored confidence == 0 are dropped.
        Returns None if input is None.
        """
        if detections is None:
            self._history.append([])
            return None

        self._history.append(detections)

        if len(self._history) < 2:
            return detections

        past_frames = list(self._history)[:-1]  # oldest → newest-1

        result: DetectionList = []
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                result.append(det)
                continue

            chain_confs = [det["confidence"]]

            # Walk backwards through history, stop when chain breaks
            for past_dets in reversed(past_frames):
                best_iou = 0.0
                best_conf: Optional[float] = None
                for pd in past_dets:
                    if pd["class_id"] != det["class_id"]:
                        continue
                    pb = pd.get("bbox")
                    if not pb or len(pb) < 4:
                        continue
                    iou = _iou(bbox, pb)
                    if iou >= self.iou_thresh and iou > best_iou:
                        best_iou = iou
                        best_conf = pd["confidence"]

                if best_conf is not None:
                    chain_confs.append(best_conf)
                else:
                    break  # chain broken — don't look further back

            chain_len = len(chain_confs)

            if chain_len >= self.min_seq:
                new_conf = max(chain_confs) if self.method == "max" else (
                    sum(chain_confs) / chain_len
                )
            else:
                new_conf = det["confidence"] * self.penalty

            if new_conf > 0.0:
                result.append({**det, "confidence": new_conf})

        return result
