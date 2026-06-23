"""
Event Detector: frame-by-frame inventory diff → purchase/return events.

State machine per class:
  UNKNOWN   → initial state; waiting for count to stabilize
  STABLE    → count confirmed (initial or post-event)
  CANDIDATE → potential event detected; awaiting confirmation

Transitions:
  UNKNOWN   + (median stable for INIT_CONFIRM frames)   → STABLE (initial count set)
  STABLE    + (median != committed, valid delta)         → CANDIDATE
  CANDIDATE + (candidate stable for CONFIRM_FRAMES)     → STABLE + event fired
  CANDIDATE + (median == committed)                     → STABLE (noise, cancelled)
  CANDIDATE + (median changes to another value)         → CANDIDATE (timer reset)

Why this approach:
  Flickering items (0↔1 every frame) never stabilize in UNKNOWN → no false events.
  Real events persist for CONFIRM_FRAMES → reliably detected.
  No MIN_EVENT_GAP needed: the confirmation window naturally prevents rapid re-triggering.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import copy
import statistics


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

WINDOW_SIZE    = 25   # sliding window size for median (odd recommended)
MAX_DELTA      = 4    # max count change allowed per event
INIT_CONFIRM   = 5    # consecutive stable frames to confirm initial inventory
CONFIRM_FRAMES = 30   # consecutive frames new state must persist to fire event
                      # skip=2 → 60 real frames ≈ 2 seconds


# ---------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------

@dataclass
class Event:
    event_num:  int
    class_id:   int
    class_name: str
    action:     str   # "구매" | "반환"
    before:     int
    after:      int
    frame_idx:  int


# ---------------------------------------------------------------
# EventDetector
# ---------------------------------------------------------------

class EventDetector:
    """
    Usage:
        detector = EventDetector(class_names, initial_counts)
        for frame_detections in video_frames:
            new_events = detector.update(frame_detections)
        events = detector.all_events
    """

    def __init__(
        self,
        class_names:    List[str],
        initial_counts: Optional[Dict[int, int]] = None,
        window_size:    int = WINDOW_SIZE,
        max_delta:      int = MAX_DELTA,
        init_confirm:   int = INIT_CONFIRM,
        confirm_frames: int = CONFIRM_FRAMES,
    ):
        self.class_names    = class_names
        self.window_size    = window_size
        self.max_delta      = max_delta
        self.init_confirm   = init_confirm
        self.confirm_frames = confirm_frames

        self.all_events: List[Event] = []
        self._event_counter = 0
        self._frame_idx     = 0

        # sliding window history per class
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

        # state machine per class: "unknown" | "stable" | "candidate"
        self._sm_state: Dict[int, str] = defaultdict(lambda: "unknown")

        # confirmed inventory per class
        self._committed: Dict[int, int] = defaultdict(int)

        # pending candidate: {cls_id: (count, since_frame)}
        self._candidate: Dict[int, Tuple[int, int]] = {}

        # classes with initial_counts provided start STABLE immediately
        if initial_counts:
            for cls_id, count in initial_counts.items():
                self._sm_state[cls_id]  = "stable"
                self._committed[cls_id] = count

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------

    def _class_name(self, cls_id: int) -> str:
        return self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

    def _median(self, cls_id: int) -> Optional[int]:
        hist = self._history[cls_id]
        if len(hist) < self.window_size:
            return None
        return round(statistics.median(hist))

    # -----------------------------------------------------------
    # Main update
    # -----------------------------------------------------------

    def update(self, detections: List[Dict]) -> List[Event]:
        """
        detections: list of {class_id: int, confidence: float, bbox: [...]}
        Returns newly fired events this frame.
        """
        frame_counts: Dict[int, int] = defaultdict(int)
        for det in detections:
            frame_counts[det["class_id"]] += 1

        new_events = []
        all_classes = set(frame_counts.keys()) | set(self._committed.keys())

        for cls_id in all_classes:
            self._history[cls_id].append(frame_counts.get(cls_id, 0))

            median = self._median(cls_id)
            if median is None:
                continue

            state = self._sm_state[cls_id]

            # ── UNKNOWN: waiting for initial inventory to stabilize ────────
            if state == "unknown":
                cand = self._candidate.get(cls_id)
                if cand is None or cand[0] != median:
                    self._candidate[cls_id] = (median, self._frame_idx)
                elif self._frame_idx - cand[1] >= self.init_confirm:
                    self._sm_state[cls_id]  = "stable"
                    self._committed[cls_id] = median
                    self._candidate.pop(cls_id, None)

            # ── STABLE: watching for changes ──────────────────────────────
            elif state == "stable":
                committed = self._committed[cls_id]
                delta = median - committed
                if delta == 0 or not (1 <= abs(delta) <= self.max_delta):
                    continue
                self._sm_state[cls_id]  = "candidate"
                self._candidate[cls_id] = (median, self._frame_idx)

            # ── CANDIDATE: waiting for event to be confirmed ───────────────
            elif state == "candidate":
                committed             = self._committed[cls_id]
                cand_count, cand_since = self._candidate[cls_id]

                if median == committed:
                    # reverted → noise, cancel
                    self._sm_state[cls_id] = "stable"
                    self._candidate.pop(cls_id, None)

                elif median == cand_count:
                    if self._frame_idx - cand_since >= self.confirm_frames:
                        # event confirmed
                        delta  = cand_count - committed
                        action = "구매" if delta < 0 else "반환"
                        after  = max(0, cand_count)

                        self._event_counter += 1
                        event = Event(
                            event_num  = self._event_counter,
                            class_id   = cls_id,
                            class_name = self._class_name(cls_id),
                            action     = action,
                            before     = committed,
                            after      = after,
                            frame_idx  = self._frame_idx,
                        )
                        self._committed[cls_id] = after
                        self._sm_state[cls_id]  = "stable"
                        self._candidate.pop(cls_id, None)
                        self.all_events.append(event)
                        new_events.append(event)

                else:
                    # candidate count changed → reset timer
                    self._candidate[cls_id] = (median, self._frame_idx)

        self._frame_idx += 1
        return new_events
