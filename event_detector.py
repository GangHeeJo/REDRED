"""
Event Detector: frame-by-frame inventory diff → purchase/return events.

Input : per-frame detection results (list of dicts per frame)
Output: list of Event objects

Event definition:
  - "purchase" : count of a class drops by ≥1
  - "return"   : count of a class rises by ≥1

Debouncing:
  - A count change must persist for CONFIRM_FRAMES consecutive frames
    before it is accepted as an event (avoids flicker from detection noise).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict, deque
import copy


CONFIRM_FRAMES = 5   # frames a count change must hold before firing


@dataclass
class Event:
    event_num: int
    class_id: int
    class_name: str
    action: str          # "구매" | "반환"
    before: int
    after: int
    frame_idx: int


class InventoryState:
    def __init__(self, initial_counts: Optional[Dict[int, int]] = None):
        self.counts: Dict[int, int] = defaultdict(int)
        if initial_counts:
            self.counts.update(initial_counts)

    def copy(self):
        new = InventoryState()
        new.counts = copy.copy(self.counts)
        return new


class EventDetector:
    """
    Usage:
        detector = EventDetector(class_names, initial_counts)
        for frame_detections in video_frames:
            new_events = detector.update(frame_detections)
        events = detector.all_events
    """

    def __init__(self, class_names: List[str],
                 initial_counts: Optional[Dict[int, int]] = None):
        self.class_names = class_names
        self.state = InventoryState(initial_counts)
        self.all_events: List[Event] = []
        self._event_counter = 0
        self._frame_idx = 0

        # Pending changes: class_id → deque of recent frame counts
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=CONFIRM_FRAMES)
        )
        # Stable counts confirmed after debounce
        self._stable: Dict[int, int] = defaultdict(int)
        if initial_counts:
            self._stable.update(initial_counts)

    def update(self, detections: List[Dict]) -> List[Event]:
        """
        detections: list of {class_id: int, confidence: float, bbox: [x1,y1,x2,y2]}
        Returns newly fired events this frame.
        """
        # Count detections per class
        frame_counts: Dict[int, int] = defaultdict(int)
        for det in detections:
            frame_counts[det["class_id"]] += 1

        new_events = []
        all_classes = set(frame_counts.keys()) | set(self._stable.keys())

        for cls_id in all_classes:
            current = frame_counts.get(cls_id, 0)
            self._history[cls_id].append(current)

            # Only fire if count has been stable for CONFIRM_FRAMES
            if len(self._history[cls_id]) < CONFIRM_FRAMES:
                continue
            if len(set(self._history[cls_id])) != 1:
                continue  # still fluctuating

            confirmed = self._history[cls_id][0]
            prev_stable = self._stable.get(cls_id, 0)

            if confirmed != prev_stable:
                action = "구매" if confirmed < prev_stable else "반환"
                self._event_counter += 1
                event = Event(
                    event_num=self._event_counter,
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    action=action,
                    before=prev_stable,
                    after=confirmed,
                    frame_idx=self._frame_idx,
                )
                self._stable[cls_id] = confirmed
                self.all_events.append(event)
                new_events.append(event)

        self._frame_idx += 1
        return new_events
