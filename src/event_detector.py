"""
Event Detector: frame-by-frame inventory diff → purchase/return events.

Input : per-frame detection results (list of dicts per frame)
Output: list of Event objects

Event definition:
  - "구매" : count of a class drops by ≥1
  - "반환" : count of a class rises by ≥1

후처리 개선 목록:
  1. Sliding window median  : 윈도우 내 중앙값으로 판단 → 단일 프레임 오탐 허용
  2. Max delta 제한         : 한 이벤트에서 수량이 MAX_DELTA 초과 변화 시 노이즈로 무시
  3. 최소 이벤트 간격       : 같은 클래스에서 MIN_EVENT_GAP 프레임 이내 재발생 무시
  4. 재고 음수 방지         : 재고가 0 미만으로 내려가면 0으로 클램핑
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict, deque
import copy
import statistics


# ---------------------------------------------------------------
# 파라미터 (필요시 EventDetector 생성자에서 override 가능)
# ---------------------------------------------------------------

WINDOW_SIZE    = 15   # sliding window 크기 (홀수 권장)
MAX_DELTA      = 4    # 한 이벤트에서 허용하는 최대 수량 변화 (초과 시 노이즈로 무시)
MIN_EVENT_GAP  = 90   # 같은 클래스에서 이벤트 재발생까지 필요한 최소 프레임 수
                      # skip=2 기준 → 실제 180프레임 ≈ 6초 (구매/반환 최소 소요시간)


# ---------------------------------------------------------------
# 데이터 클래스
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


class InventoryState:
    def __init__(self, initial_counts: Optional[Dict[int, int]] = None):
        self.counts: Dict[int, int] = defaultdict(int)
        if initial_counts:
            self.counts.update(initial_counts)

    def copy(self):
        new = InventoryState()
        new.counts = copy.copy(self.counts)
        return new


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

    Args:
        class_names    : 60개 클래스 이름 리스트
        initial_counts : {class_id: 초기 재고} — run_pipeline의 estimate_initial_inventory 결과
        window_size    : sliding window 크기 (기본 7)
        max_delta      : 이벤트 1건당 허용 최대 수량 변화 (기본 4)
        min_event_gap  : 같은 클래스 이벤트 재발생 최소 프레임 간격 (기본 10)
    """

    def __init__(
        self,
        class_names:    List[str],
        initial_counts: Optional[Dict[int, int]] = None,
        window_size:    int = WINDOW_SIZE,
        max_delta:      int = MAX_DELTA,
        min_event_gap:  int = MIN_EVENT_GAP,
    ):
        self.class_names   = class_names
        self.window_size   = window_size
        self.max_delta     = max_delta
        self.min_event_gap = min_event_gap

        self.state = InventoryState(initial_counts)
        self.all_events: List[Event] = []
        self._event_counter = 0
        self._frame_idx     = 0

        # [개선 1] sliding window: class_id → deque of recent raw counts
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

        # debounce 확정된 안정 카운트
        self._stable: Dict[int, int] = defaultdict(int)
        if initial_counts:
            self._stable.update(initial_counts)

        # [개선 3] 마지막 이벤트 발생 프레임: class_id → frame_idx
        self._last_event_frame: Dict[int, int] = {}

    # -----------------------------------------------------------
    # 내부 헬퍼
    # -----------------------------------------------------------

    def _class_name(self, cls_id: int) -> str:
        if cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return f"class_{cls_id}"

    def _sliding_median(self, cls_id: int) -> Optional[int]:
        """
        [개선 1] sliding window median.
        윈도우가 꽉 찬 경우에만 중앙값 반환, 아직 쌓이는 중이면 None.
        중앙값이 정수가 되도록 반올림.
        """
        hist = self._history[cls_id]
        if len(hist) < self.window_size:
            return None
        return round(statistics.median(hist))

    def _is_valid_delta(self, delta: int) -> bool:
        """[개선 2] 수량 변화가 MAX_DELTA 이내인지 확인."""
        return 1 <= abs(delta) <= self.max_delta

    def _is_gap_ok(self, cls_id: int) -> bool:
        """[개선 3] 마지막 이벤트로부터 MIN_EVENT_GAP 프레임 이상 지났는지 확인."""
        last = self._last_event_frame.get(cls_id, -self.min_event_gap)
        return (self._frame_idx - last) >= self.min_event_gap

    # -----------------------------------------------------------
    # 메인 업데이트
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
        all_classes = set(frame_counts.keys()) | set(self._stable.keys())

        for cls_id in all_classes:
            current = frame_counts.get(cls_id, 0)
            self._history[cls_id].append(current)

            # [개선 1] 중앙값으로 안정 카운트 판단
            median_count = self._sliding_median(cls_id)
            if median_count is None:
                continue

            prev_stable = self._stable.get(cls_id, 0)
            delta = median_count - prev_stable

            if delta == 0:
                continue

            # [개선 2] 비정상적으로 큰 변화는 노이즈로 무시
            if not self._is_valid_delta(delta):
                continue

            # [개선 3] 최소 이벤트 간격 확인
            if not self._is_gap_ok(cls_id):
                continue

            action = "구매" if delta < 0 else "반환"

            # [개선 4] 재고 음수 방지: 구매 후 재고가 0 미만이면 클램핑
            after = max(0, median_count)

            self._event_counter += 1
            event = Event(
                event_num  = self._event_counter,
                class_id   = cls_id,
                class_name = self._class_name(cls_id),
                action     = action,
                before     = prev_stable,
                after      = after,
                frame_idx  = self._frame_idx,
            )

            self._stable[cls_id]           = after
            self._last_event_frame[cls_id] = self._frame_idx
            self.all_events.append(event)
            new_events.append(event)

        self._frame_idx += 1
        return new_events
