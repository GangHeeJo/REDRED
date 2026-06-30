"""
Event Detector: frame-by-frame inventory diff → purchase/return events.

State machine per class:
  STABLE    → default initial state; committed=0 unless initial_counts provided
  CANDIDATE → potential event detected; awaiting confirmation

Transitions:
  STABLE    + (median != committed, valid delta)         → CANDIDATE
  CANDIDATE + (candidate stable for CONFIRM_FRAMES)     → STABLE + event fired
  CANDIDATE + (median == committed)                     → STABLE (noise, cancelled)
  CANDIDATE + (median changes to another value)         → CANDIDATE (timer reset)

All classes default to STABLE with committed=0. This ensures items absent at
video start (initial count=0) can correctly detect return events (0→1) when
they first appear, instead of confirming the wrong initial state via UNKNOWN.

Inventory constraints (physical limits):
  - Return blocked if committed >= MAX_INVENTORY  (shelf is full)
  Purchase-from-empty is NOT blocked — initial inventory estimation is
  unreliable enough that blocking causes more FN than it prevents FP.
  CONFIRM_FRAMES already filters noise sufficiently.

After each event, the sliding window history for that class is cleared.
This forces a fresh 25-frame fill before the next candidate can form,
acting as a natural cooldown that prevents re-triggering on residual noise.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import copy
import statistics


# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------

WINDOW_SIZE    = 15   # sliding window size for median (odd recommended)
MAX_DELTA      = 4    # max count change allowed per event
CONFIRM_FRAMES = 30   # consecutive frames new state must persist to fire event
                      # skip=2 → 60 real frames ≈ 2 seconds
MAX_INVENTORY  = 1    # physical shelf capacity per slot (most items: 0↔1)


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
        confirm_frames: int = CONFIRM_FRAMES,
        max_inventory:  int = MAX_INVENTORY,
        per_class_confirm: Optional[Dict[int, int]] = None,
        use_ema:        bool = False,
        ema_alpha:      float = 0.3,
    ):
        self.class_names    = class_names
        self.window_size    = window_size
        self.max_delta      = max_delta
        self.confirm_frames = confirm_frames
        self.max_inventory  = max_inventory
        self.per_class_confirm = per_class_confirm or {}
        self.use_ema        = use_ema
        self.ema_alpha      = ema_alpha

        self.all_events: List[Event] = []
        self._event_counter = 0
        self._frame_idx     = 0

        # sliding window history per class (median mode)
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

        # EMA values per class (ema mode)
        self._ema_vals: Dict[int, float] = {}
        self._ema_frames: Dict[int, int] = {}   # frames seen (warmup counter)

        # state machine per class: "stable" | "candidate"
        self._sm_state: Dict[int, str] = defaultdict(lambda: "stable")

        # confirmed inventory per class
        self._committed: Dict[int, int] = defaultdict(int)

        # pending candidate: {cls_id: (count, since_frame)}
        self._candidate: Dict[int, Tuple[int, int]] = {}

        # classes with initial_counts provided start STABLE immediately
        if initial_counts:
            for cls_id, count in initial_counts.items():
                self._sm_state[cls_id]  = "stable"
                self._committed[cls_id] = count
                self._ema_vals[cls_id]   = float(count)
                self._ema_frames[cls_id] = 3  # already warmed up

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

    def _ema(self, cls_id: int, raw_count: int) -> Optional[int]:
        """EMA update: returns rounded EMA after min 3-frame warmup."""
        α = self.ema_alpha
        if cls_id not in self._ema_vals:
            self._ema_vals[cls_id]  = float(raw_count)
            self._ema_frames[cls_id] = 1
            return None  # warmup
        self._ema_vals[cls_id]   = α * raw_count + (1 - α) * self._ema_vals[cls_id]
        self._ema_frames[cls_id] += 1
        if self._ema_frames[cls_id] < 3:
            return None  # warmup
        return round(self._ema_vals[cls_id])

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
            raw = frame_counts.get(cls_id, 0)
            self._history[cls_id].append(raw)

            if self.use_ema:
                median = self._ema(cls_id, raw)
            else:
                median = self._median(cls_id)
            if median is None:
                continue

            state = self._sm_state[cls_id]

            # ── STABLE: watching for changes ──────────────────────────────
            if state == "stable":
                committed = self._committed[cls_id]
                delta = median - committed
                if delta == 0 or not (1 <= abs(delta) <= self.max_delta):
                    continue
                self._sm_state[cls_id]  = "candidate"
                self._candidate[cls_id] = (median, self._frame_idx)

            # ── CANDIDATE: waiting for event to be confirmed ───────────────
            else:  # state == "candidate"
                committed             = self._committed[cls_id]
                cand_count, cand_since = self._candidate[cls_id]

                if median == committed:
                    # reverted → noise, cancel
                    self._sm_state[cls_id] = "stable"
                    self._candidate.pop(cls_id, None)

                elif median == cand_count:
                    cf = self.per_class_confirm.get(cls_id, self.confirm_frames)
                    if self._frame_idx - cand_since >= cf:
                        # event confirmed — check physical constraints
                        delta  = cand_count - committed
                        action = "구매" if delta < 0 else "반환"

                        # return to full shelf: impossible
                        if action == "반환" and committed >= self.max_inventory:
                            self._sm_state[cls_id] = "stable"
                            self._candidate.pop(cls_id, None)
                            continue

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
                        # reset history so next event needs a fresh window
                        self._history[cls_id].clear()
                        self.all_events.append(event)
                        new_events.append(event)

                else:
                    # candidate count changed → reset timer
                    self._candidate[cls_id] = (median, self._frame_idx)

        self._frame_idx += 1
        return new_events
