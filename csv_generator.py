"""
CSV Generator: converts Event list -> competition submission CSV.

Output format per row by default:
    품목명, 이벤트 번호, 이벤트 후 재고 수량, 총액

Example:
    Pringles_bbq, Event 3, 재고 수량: 7개, 총액: 10500원

Expected event fields:
    event.class_id
    event.event_num
    event.before
    event.after
    event.action      # optional, "구매" or "반환"

Also supports dictionary events:
    {
        "class_id": 0,
        "event_num": 7,
        "before": 3,
        "after": 2,
        "action": "구매"
    }

prices.csv format:
    class_id, class_name, price_krw   (preferred)
    class_id, class_name, price_usd   (fallback)

Main changes:
    1. [수정] event_detector.Event import 의존성 제거
    2. [수정] action을 event.action만 믿지 않고 before/after 차이로 재계산
    3. [수정] initial_inventory를 실제 검증에 사용
    4. [수정] 총액 계산 기준을 선택 가능하게 변경
        - total_mode="global"    : 전체 누적 총액
        - total_mode="per_class" : 상품별 누적 총액, 기존 스켈레톤 방식
        - total_mode="event"     : 해당 이벤트 금액
    5. [수정] unknown class, no-change event, inventory mismatch 경고 처리
    6. [추가] Dict event와 dataclass/object event 모두 지원
    7. [추가] 가격 float 대신 Decimal 사용
    8. [추가] 공식 제출 형식에 맞추기 위해 include_action 옵션 제공
"""

import csv
import os
import re
import warnings
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


PriceInfo = Tuple[str, Decimal]
RawPriceInfo = Union[Tuple[str, Any], Mapping[str, Any]]
EventLike = Any


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

def _get(event: EventLike, *names: str, default: Any = None) -> Any:
    """
    [추가]
    Event가 object/dataclass여도 되고 dict여도 되게 만드는 helper.

    Example:
        _get(event, "event_num", "event_id", default=0)
    """
    if isinstance(event, Mapping):
        for name in names:
            if name in event:
                return event[name]
        return default

    for name in names:
        if hasattr(event, name):
            return getattr(event, name)

    return default


def _to_int(value: Any, field_name: str) -> int:
    """
    [추가]
    class_id, before, after 등을 안전하게 int로 변환.
    """
    if value is None:
        raise ValueError(f"{field_name} is missing")

    text = str(value).strip()

    if text == "":
        raise ValueError(f"{field_name} is empty")

    try:
        return int(float(text))
    except ValueError as e:
        raise ValueError(f"{field_name} must be int-like, got {value!r}") from e


def _clean_money_text(value: Any) -> str:
    """
    [추가]
    "$3.50", "3.50", "1,200", "1200원" 같은 값도 Decimal로 바꿀 수 있게 정리.
    """
    text = str(value).strip()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("USD", "")
    text = text.replace("usd", "")
    text = text.replace("원", "")
    return text.strip()


def _to_decimal(value: Any) -> Decimal:
    """
    [수정]
    float 대신 Decimal 사용.
    돈 계산은 float보다 Decimal이 안전함.
    """
    text = _clean_money_text(value)

    if text == "":
        return Decimal("0")

    return Decimal(text)


def _money_quant(money_digits: int) -> Decimal:
    """
    [추가]
    money_digits=2 -> Decimal("0.01")
    money_digits=1 -> Decimal("0.1")
    """
    if money_digits <= 0:
        return Decimal("1")

    return Decimal("0." + "0" * (money_digits - 1) + "1")


def format_money(
    amount: Decimal,
    currency: str = "원",
    money_digits: int = 0,
    currency_as_suffix: bool = True,
) -> str:
    """
    [수정]
    총액 출력 포맷 통일.

    Example:
        Decimal("10500"), currency="원", suffix=True  -> "10500원"
        Decimal("23.5"),  currency="$",  suffix=False -> "$23.50"
    """
    quant = _money_quant(money_digits)
    rounded = amount.quantize(quant, rounding=ROUND_HALF_UP)
    formatted = f"{rounded:.{money_digits}f}"
    if currency_as_suffix:
        return f"{formatted}{currency}"
    return f"{currency}{formatted}"


# ------------------------------------------------------------
# Price loading
# ------------------------------------------------------------

def _find_field(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    """
    [추가]
    CSV header가 조금 달라도 읽을 수 있도록 field 이름 탐색.
    """
    normalized = {
        name.strip().lower(): name
        for name in fieldnames
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def load_prices(prices_csv: str) -> Dict[int, PriceInfo]:
    """
    prices.csv format:
        class_id, class_name, price_usd

    Returns:
        {
            class_id: (class_name, price_usd_as_Decimal)
        }

    [수정]
    - utf-8-sig로 읽어서 Excel에서 만든 CSV의 BOM 문제 대응
    - price_usd 대신 price, 가격 같은 header도 일부 허용
    - 가격을 float이 아니라 Decimal로 저장
    """
    prices: Dict[int, PriceInfo] = {}

    with open(prices_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"prices CSV has no header: {prices_csv}")

        fieldnames = reader.fieldnames

        id_field = _find_field(
            fieldnames,
            ["class_id", "cls_id", "id", "class"]
        )
        name_field = _find_field(
            fieldnames,
            ["class_name", "name", "product_name", "product", "품목명"]
        )
        price_field = _find_field(
            fieldnames,
            ["price_krw", "price_usd", "price", "krw", "usd", "가격"]
        )

        if id_field is None:
            raise ValueError("prices CSV must have class_id column")

        if name_field is None:
            raise ValueError("prices CSV must have class_name column")

        if price_field is None:
            raise ValueError("prices CSV must have price_usd or price column")

        for row_idx, row in enumerate(reader, start=2):
            if not row.get(id_field):
                continue

            try:
                cls_id = _to_int(row[id_field], "class_id")
                class_name = str(row[name_field]).strip()
                price_usd = _to_decimal(row[price_field])
            except Exception as e:
                warnings.warn(f"[WARN] skip invalid price row {row_idx}: {e}")
                continue

            if not class_name:
                class_name = f"class_{cls_id}"

            if cls_id in prices:
                warnings.warn(f"[WARN] duplicated class_id in prices.csv: {cls_id}")

            prices[cls_id] = (class_name, price_usd)

    if not prices:
        raise ValueError(f"no valid price rows loaded from: {prices_csv}")

    return prices


def normalize_prices(prices: Mapping[int, RawPriceInfo]) -> Dict[int, PriceInfo]:
    """
    [추가]
    load_prices() 결과뿐 아니라 사용자가 직접 만든 가격 dict도 받을 수 있게 정규화.

    Supported:
        {
            0: ("a1_steak_sauce", 3.5)
        }

        {
            0: {"class_name": "a1_steak_sauce", "price_usd": 3.5}
        }
    """
    normalized: Dict[int, PriceInfo] = {}

    for raw_cls_id, value in prices.items():
        cls_id = int(raw_cls_id)

        if isinstance(value, Mapping):
            name = next(
                (value.get(k) for k in ("class_name", "name", "product_name") if value.get(k)),
                f"class_{cls_id}",
            )
            price = next(
                (value.get(k) for k in ("price_krw", "price_usd", "price", "krw", "usd") if value.get(k) is not None),
                0,
            )
        else:
            name, price = value

        normalized[cls_id] = (str(name), _to_decimal(price))

    return normalized


# ------------------------------------------------------------
# Event handling
# ------------------------------------------------------------

def _normalize_action(action: Any) -> Optional[str]:
    """
    [추가]
    action 문자열을 "구매" 또는 "반환"으로 정규화.
    """
    if action is None:
        return None

    text = str(action).strip().lower()

    buy_words = {"구매", "buy", "bought", "purchase", "purchased", "take", "taken"}
    return_words = {"반환", "return", "returned", "refund", "putback", "put_back"}

    if text in buy_words:
        return "구매"

    if text in return_words:
        return "반환"

    return None


def derive_action(before: int, after: int, provided_action: Any = None) -> str:
    """
    [수정]
    event.action만 믿지 않고 before/after 차이를 기준으로 구매/반환을 판단.

    before > after:
        상품 수량 감소 -> 구매

    before < after:
        상품 수량 증가 -> 반환

    before == after:
        변화없음
    """
    delta = after - before

    if delta < 0:
        derived = "구매"
    elif delta > 0:
        derived = "반환"
    else:
        derived = "변화없음"

    provided = _normalize_action(provided_action)

    if provided is not None and provided != derived and derived != "변화없음":
        warnings.warn(
            f"[WARN] provided action={provided!r} but before/after imply {derived!r}. "
            f"Use derived action."
        )

    return derived


def _event_num_to_int(event_num: Any) -> int:
    """
    [추가]
    정렬용 event number 추출.
    'Event 7' -> 7
    7 -> 7
    """
    if event_num is None:
        return 0

    match = re.search(r"\d+", str(event_num))

    if match:
        return int(match.group())

    return 0


def format_event_label(event_num: Any) -> str:
    """
    [수정]
    event_num이 7이면 'Event 7',
    이미 'Event 7'이면 그대로 사용.
    """
    if event_num is None:
        return "Event unknown"

    text = str(event_num).strip()

    if text == "":
        return "Event unknown"

    if text.lower().startswith("event"):
        return text

    return f"Event {text}"


# ------------------------------------------------------------
# CSV generation
# ------------------------------------------------------------

def events_to_csv(
    events: Iterable[EventLike],
    prices: Mapping[int, RawPriceInfo],
    out_path: str,
    initial_inventory: Optional[Mapping[int, int]] = None,
    total_mode: str = "global",
    include_action: bool = False,
    include_delta: bool = False,
    skip_no_change: bool = True,
    sort_events: bool = True,
    currency: str = "원",
    money_digits: int = 0,
    currency_as_suffix: bool = True,
    encoding: str = "utf-8",
) -> List[List[str]]:
    """
    Generates the submission CSV.

    Args:
        events:
            Event objects or dictionaries.
            Required fields:
                class_id, before, after
            Recommended fields:
                event_num, action

        prices:
            {class_id: (class_name, price_usd)}
            or
            {class_id: {"class_name": ..., "price_usd": ...}}

        out_path:
            output CSV path.

        initial_inventory:
            {class_id: initial_stock} before any events.
            [수정] 기존 스켈레톤에서는 받기만 하고 안 썼는데,
            이제 event.before와 비교하는 검증에 사용함.

        total_mode:
            [수정] 총액 계산 기준 선택.

            "global":
                전체 상품 기준 누적 총액.
                대회에서 "총 판매 금액"이라고 하면 이 방식이 가장 자연스러움.

            "per_class":
                상품별 누적 총액.
                기존 네 스켈레톤 코드와 가장 가까운 방식.

            "event":
                해당 이벤트 한 건의 금액.
                반환이면 음수 금액으로 출력될 수 있음.

        include_action:
            True면 "구매/반환 여부" column 포함.
            공식 제출 형식에서 이 column이 필요 없다면 False로 바꾸면 됨.

        include_delta:
            True면 "변화량" column 추가.
            디버깅용으로 좋지만, 공식 제출 때는 보통 False 권장.

        skip_no_change:
            before == after인 event를 CSV에서 제외할지 여부.

        sort_events:
            Event 번호 순서대로 정렬할지 여부.

        currency:
            "원" (기본) or "$" 등.

        money_digits:
            총액 소수점 자리수.
            예: 0 -> 10500원  (기본, Korean won)
                2 -> $23.50

        currency_as_suffix:
            True면 "10500원", False면 "$23.50"

        encoding:
            기본은 utf-8.
            Excel에서 한글이 깨지면 "utf-8-sig"로 바꿔도 됨.

    Returns:
        rows written to CSV, excluding header.
    """
    if total_mode not in {"global", "per_class", "event"}:
        raise ValueError(
            "total_mode must be one of: 'global', 'per_class', 'event'"
        )

    price_map = normalize_prices(prices)

    inventory: Dict[int, int] = {}
    if initial_inventory is not None:
        inventory = {
            int(cls_id): int(stock)
            for cls_id, stock in initial_inventory.items()
        }

    event_list = list(events)

    # [추가] event 번호 기준 정렬
    if sort_events:
        def _sort_key(ev):
            num = _event_num_to_int(_get(ev, "event_num", "event_id", "event", default=0))
            try:
                cls = _to_int(_get(ev, "class_id", "cls_id", default=None), "class_id")
            except (ValueError, TypeError):
                cls = 999999
            return (num, cls)

        event_list.sort(key=_sort_key)

    global_total = Decimal("0")
    class_total: Dict[int, Decimal] = {}

    header = ["품목명", "이벤트 번호"]

    if include_action:
        header.append("구매/반환 여부")

    if include_delta:
        header.append("변화량")

    header += ["이벤트 후 재고 수량", "총액"]

    rows: List[List[str]] = []
    skipped_count = 0

    for idx, event in enumerate(event_list, start=1):
        try:
            cls_id = _to_int(
                _get(event, "class_id", "cls_id", "id", default=None),
                "class_id"
            )
            before = _to_int(
                _get(event, "before", "before_count", "before_stock", default=None),
                "before"
            )
            after = _to_int(
                _get(event, "after", "after_count", "after_stock", default=None),
                "after"
            )
        except Exception as e:
            warnings.warn(f"[WARN] skip invalid event index={idx}: {e}")
            skipped_count += 1
            continue

        if cls_id not in price_map:
            warnings.warn(f"[WARN] skip unknown class_id={cls_id}")
            skipped_count += 1
            continue

        event_num = _get(
            event,
            "event_num",
            "event_id",
            "event",
            default=idx
        )
        event_label = format_event_label(event_num)

        provided_action = _get(
            event,
            "action",
            "event_type",
            "type",
            default=None
        )

        action = derive_action(before, after, provided_action)
        delta = after - before

        if delta == 0 and skip_no_change:
            skipped_count += 1
            continue

        # [수정] initial_inventory와 event.before가 맞는지 확인
        if cls_id not in inventory:
            inventory[cls_id] = before
        else:
            expected_before = inventory[cls_id]
            if expected_before != before:
                warnings.warn(
                    f"[WARN] inventory mismatch at {event_label}, class_id={cls_id}: "
                    f"running inventory says before={expected_before}, "
                    f"event says before={before}. "
                    f"Use event.after={after} for next inventory."
                )

        # [수정] 다음 이벤트 검증을 위해 현재 after로 재고 업데이트
        inventory[cls_id] = after

        name, price = price_map[cls_id]
        change_count = abs(delta)

        if cls_id not in class_total:
            class_total[cls_id] = Decimal("0")

        # [수정] 총액 계산
        if action == "구매":
            event_amount = price * Decimal(change_count)
            global_total += event_amount
            class_total[cls_id] += event_amount

        elif action == "반환":
            # 반환은 환불/판매취소로 보고 누적 총액에서 차감
            event_amount = -(price * Decimal(change_count))
            global_total += event_amount
            class_total[cls_id] += event_amount

            # 총액이 음수가 되지 않게 방어
            if global_total < Decimal("0"):
                global_total = Decimal("0")

            if class_total[cls_id] < Decimal("0"):
                class_total[cls_id] = Decimal("0")

        else:
            event_amount = Decimal("0")

        if total_mode == "global":
            total_to_write = global_total
        elif total_mode == "per_class":
            total_to_write = class_total[cls_id]
        else:
            total_to_write = event_amount

        row = [
            name,
            event_label,
        ]

        if include_action:
            row.append(action)

        if include_delta:
            # after - before
            # 구매면 음수, 반환이면 양수
            row.append(f"{delta:+d}")

        row += [
            f"재고 수량: {after}개",
            f"총액: {format_money(total_to_write, currency=currency, money_digits=money_digits, currency_as_suffix=currency_as_suffix)}",
        ]

        rows.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows -> {out_path}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} events")

    return rows


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    """
    간단 테스트용 예시.

    실제 사용 시에는 event_detector.py에서 만든 events를 넘기면 됨.
    """

    # [추가] dict event도 지원됨
    sample_events = [
        {
            "class_id": 0,
            "event_num": 1,
            "before": 5,
            "after": 4,
            "action": "구매",
        },
        {
            "class_id": 1,
            "event_num": 2,
            "before": 3,
            "after": 5,
            "action": "반환",
        },
    ]

    # [추가] prices.csv 없이도 테스트 가능 (단위: 원)
    sample_prices = {
        0: ("a1_steak_sauce", Decimal("4980")),
        1: ("Pringles_bbq", Decimal("1500")),
    }

    sample_initial_inventory = {
        0: 5,
        1: 8,
    }

    events_to_csv(
        events=sample_events,
        prices=sample_prices,
        out_path="submission_draft.csv",
        initial_inventory=sample_initial_inventory,

        # "global"    : 전체 누적 총액
        # "per_class" : 기존 스켈레톤처럼 상품별 누적 총액
        # "event"     : 해당 이벤트 금액
        total_mode="global",

        include_action=False,   # 제출 포맷 기준
        include_delta=False,
        currency="원",
        money_digits=0,
        currency_as_suffix=True,
    )