"""
Ghost Detector: fills missed detections using last-known bbox positions.

Inspired by:
  "Objects Do Not Disappear: Video Object Detection by
   Single-Frame Object Location Anticipation"
  Liu et al., ICCV 2023 — https://arxiv.org/abs/2308.04770

Core idea: objects in video don't vanish instantaneously. When the model
temporarily loses a detection (e.g. hand occlusion, lighting change), the
object is almost certainly still at its last known position. We inject a
"ghost" detection at that position with decaying confidence until either
the object is re-detected or the ghost expires.

REDRED motivation:
  dove_white: YOLO11m loses detection at 78s, GT purchase at 105s → 27s gap
  milano: loses detection at 67s, GT purchase at 115s → 48s gap
  Fix: per-class ghost fills the gap → EventDetector sees count=1 until
       the item is actually taken.

Usage (per camera, applied before multi-view fusion):
    gd = GhostDetector(
        max_frames=30,
        conf_decay=1.0,
        min_conf=0.3,
        per_class_max_frames={42: 700, 54: 450},  # milano, dove_white
    )
    per_cam_dets = gd.update(per_cam_dets)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple


DetectionList = List[Dict]


class GhostDetector:
    """
    Per-camera ghost detection filler.

    Parameters
    ----------
    max_frames          : default ghost lifetime (processed frames)
    conf_decay          : confidence multiplier per ghost frame (1.0 = no decay)
    min_conf            : ghost expires when decayed conf drops below this
    per_class_max_frames: {class_id: max_frames} overrides per class
    per_class_conf_decay: {class_id: decay} overrides per class
    """

    def __init__(
        self,
        max_frames: int = 30,
        conf_decay: float = 1.0,
        min_conf: float = 0.3,
        per_class_max_frames: Optional[Dict[int, int]] = None,
        per_class_conf_decay: Optional[Dict[int, float]] = None,
    ) -> None:
        self.max_frames           = max_frames
        self.conf_decay           = conf_decay
        self.min_conf             = min_conf
        self.per_class_max_frames = per_class_max_frames or {}
        self.per_class_conf_decay = per_class_conf_decay or {}

        # _registry[cam_id][cls_id] = {"bboxes": [...], "conf": float, "age": int}
        self._registry: Dict[int, Dict[int, Dict]] = {}

    def reset(self) -> None:
        self._registry.clear()

    def update(
        self, per_cam_dets: List[Optional[DetectionList]]
    ) -> List[Optional[DetectionList]]:
        """
        Apply ghost filling to all cameras for one processed frame.

        Returns a new per_cam_dets list with ghost detections inserted
        for classes that were recently seen but are missing this frame.
        Ghost detections are marked with _ghost=True for debugging.
        """
        result: List[Optional[DetectionList]] = []

        for cam_id, dets in enumerate(per_cam_dets):
            if dets is None:
                result.append(None)
                continue

            if cam_id not in self._registry:
                self._registry[cam_id] = {}
            cam_reg = self._registry[cam_id]

            # ── Group current detections by class ──────────────────────
            seen: Dict[int, List[Dict]] = {}
            for det in dets:
                seen.setdefault(det["class_id"], []).append(det)

            # ── Update registry for detected classes ───────────────────
            for cls_id, cls_dets in seen.items():
                best_conf = max(d["confidence"] for d in cls_dets)
                cam_reg[cls_id] = {
                    "bboxes": [d["bbox"] for d in cls_dets if d.get("bbox")],
                    "conf":   best_conf,
                    "age":    0,
                }

            # ── Inject ghosts for missing classes ──────────────────────
            new_dets = list(dets)
            for cls_id in list(cam_reg.keys()):
                if cls_id in seen:
                    continue  # real detection — no ghost needed

                ghost = cam_reg[cls_id]
                ghost["age"] += 1

                mf    = self.per_class_max_frames.get(cls_id, self.max_frames)
                decay = self.per_class_conf_decay.get(cls_id, self.conf_decay)
                ghost_conf = ghost["conf"] * (decay ** ghost["age"])

                if ghost["age"] > mf or ghost_conf < self.min_conf:
                    del cam_reg[cls_id]
                    continue

                for bbox in ghost["bboxes"]:
                    new_dets.append({
                        "class_id":   cls_id,
                        "confidence": ghost_conf,
                        "bbox":       bbox,
                        "_ghost":     True,
                    })

            result.append(new_dets)

        return result
