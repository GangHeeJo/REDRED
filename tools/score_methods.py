"""
Three complementary ways to score a submission against ground truth.
Each has a different blind spot, found the hard way during the 2026-06-23
accuracy debugging session (see PROGRESS.md Phase 7) -- run all three
together rather than trusting any single number in isolation.

  1. count  - per (class_name, action) frequency only (Counter). Ignores
              order/time entirely. Fast, simple, but can give false credit:
              if an event fires at a wildly wrong time but the class+action
              totals happen to balance out elsewhere, this method won't
              catch it (see pop_tararts_strawberry: looked fine by count,
              was actually firing 75s away from the real event).

  2. order  - longest common subsequence (LCS) between the two FULL
              chronological sequences of (class_name, action), with no
              per-class grouping. This is the most direct way to ask
              "is the relative order of events -- across all classes --
              right", without needing literal timestamps.

  3. time   - direct comparison of each matched event's actual fire time
              vs ground truth time, after subtracting the system's median
              detection delay (CONFIRM_FRAMES introduces a structural lag,
              currently ~2.8s). The strictest check; requires real time_sec
              on both sides.

Usage:
    python tools/score_methods.py \
        --gt data/ground_truth_v2_fixed.csv \
        --sub output/submission_skip2.csv \
        --timed output/submission_skip2_timed.csv

`--timed` (from `run_pipeline.py --timed_log`) is only needed for method 3.
If omitted, methods 1 and 2 still run using `--sub`.

`--gt` must have a `time_sec` column with real seconds.
"""

import argparse
import csv
from collections import Counter, defaultdict


ACTION_MAP = {"구매": "purchase", "반환": "return"}


def load_gt(path):
    """Returns list of (time_sec, class_name, action), sorted by time."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["time_sec"]), r["class_name"].strip(), r["action"].strip()))
    rows.sort(key=lambda x: x[0])
    return rows


def load_submission(path):
    """Returns list of (class_name, action) in submission row order (== chronological,
    since event_num is assigned in firing order)."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = r.get("품목명", "").strip()
            action = ACTION_MAP.get(r.get("구매/반환 여부", "").strip(), r.get("구매/반환 여부", "").strip())
            if name and action:
                rows.append((name, action))
    return rows


def load_timed(path):
    """Returns list of (time_sec, class_name, action), sorted by time."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            action = ACTION_MAP.get(r["action"].strip(), r["action"].strip())
            rows.append((float(r["time_sec"]), r["class_name"].strip(), action))
    rows.sort(key=lambda x: x[0])
    return rows


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def print_score(label, tp, fp, fn):
    precision, recall, f1 = prf1(tp, fp, fn)
    print(f"  [{label}] TP={tp} FP={fp} FN={fn}  "
          f"Precision={precision*100:.1f}%  Recall={recall*100:.1f}%  F1={f1*100:.1f}%")
    return f1


# ── Method 1: count only ────────────────────────────────────────────────

def score_count(gt_keys, sub_keys):
    gt_counter = Counter(gt_keys)
    sub_counter = Counter(sub_keys)
    all_keys = set(gt_counter) | set(sub_counter)

    tp = fp = fn = 0
    details = []
    for key in all_keys:
        g, s = gt_counter[key], sub_counter[key]
        tp += min(g, s)
        fp += max(0, s - g)
        fn += max(0, g - s)
        if g != s:
            details.append((key[0], key[1], g, s))
    return tp, fp, fn, details


# ── Method 2: global class-agnostic order (LCS) ────────────────────────

def score_order(gt_seq, sub_seq):
    n, m = len(gt_seq), len(sub_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_seq[i - 1] == sub_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    tp = dp[n][m]
    return tp, m - tp, n - tp


# ── Method 3: time-value matching with delay-bias correction ───────────

def estimate_bias(gt_timed, sub_timed, probe_window=10.0):
    gt_by_key = defaultdict(list)
    for t, cls, action in gt_timed:
        gt_by_key[(cls, action)].append(t)
    sub_by_key = defaultdict(list)
    for t, cls, action in sub_timed:
        sub_by_key[(cls, action)].append(t)

    diffs = []
    for key in set(gt_by_key) | set(sub_by_key):
        g, s = sorted(gt_by_key.get(key, [])), sorted(sub_by_key.get(key, []))
        for i in range(min(len(g), len(s))):
            d = s[i] - g[i]
            if abs(d) <= probe_window:
                diffs.append(d)
    if not diffs:
        return 0.0
    diffs.sort()
    return diffs[len(diffs) // 2]


def _match_nearest(g, s, bias, tolerance):
    """
    g, s: 정렬된 시각 리스트 (같은 (class,action) 키).
    인덱스로 그냥 짝짓지 않고, 각 GT 이벤트를 아직 안 쓴 Sub 이벤트 중 가장
    시각이 가까운 것과 짝지음 (그리디 nearest-available). 중복발화(Sub가 GT보다
    많음)로 인해 진짜 맞는 이벤트가 엉뚱한 occurrence와 인덱스로 강제 매칭되는
    문제를 막음 -- 2026-07 REDRED RF-DETR 세션에서 발견됨: 순수 인덱스 매칭은
    Sub=3인데 GT=1인 클래스에서 실제로는 거의 정확한 이벤트가 있어도 항상 가장
    이른(대개 유령) occurrence와 짝지어져 거대한 diff로 오판됨.

    인과성 제약: 이 파이프라인은 항상 실제 이벤트보다 "늦게" 감지함
    (CONFIRM_FRAMES 지연, bias는 그 중앙값). bias 보정 후에도 Sub가 GT보다
    tolerance 이상 이르면 그건 같은 이벤트의 지연 감지가 아니라 애초에 다른
    이벤트(유령/오탐)일 가능성이 높음 -- 억지로 매칭하지 않고 그 GT는 FN으로,
    그 Sub는 별도 FP로 남김. GT 기록 자체도 완벽하진 않아서(ground_truth_v2
    제작 중 기록 오류가 실제로 있었음) tolerance만큼의 이른 마진은 허용.
    Returns: pairs=[(gt_t, sub_t, diff), ...], unmatched_gt_count, unmatched_sub_count
    """
    used = [False] * len(s)
    pairs = []
    for gt_t in g:
        best_j, best_d = None, None
        for j, st in enumerate(s):
            if used[j]:
                continue
            raw_diff = (st - bias) - gt_t
            if raw_diff < -tolerance:
                continue  # causally implausible: 감지가 GT보다 마진 이상 이름
            d = abs(raw_diff)
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used[best_j] = True
            pairs.append((gt_t, s[best_j], best_d))
    unmatched_sub = len(s) - sum(used)
    unmatched_gt = len(g) - len(pairs)
    return pairs, unmatched_gt, unmatched_sub


def score_time(gt_timed, sub_timed, tolerance=3.0, bias=None):
    if bias is None:
        bias = estimate_bias(gt_timed, sub_timed)

    gt_by_key = defaultdict(list)
    for t, cls, action in gt_timed:
        gt_by_key[(cls, action)].append(t)
    sub_by_key = defaultdict(list)
    for t, cls, action in sub_timed:
        sub_by_key[(cls, action)].append(t)

    tp = fp = fn = 0
    mismatches = []
    for key in set(gt_by_key) | set(sub_by_key):
        g, s = sorted(gt_by_key.get(key, [])), sorted(sub_by_key.get(key, []))
        pairs, unmatched_gt, unmatched_sub = _match_nearest(g, s, bias, tolerance)
        for gt_t, sub_t, diff in pairs:
            if diff <= tolerance:
                tp += 1
            else:
                fp += 1
                fn += 1
                mismatches.append((key[0], key[1], gt_t, sub_t, diff))
        fn += unmatched_gt
        fp += unmatched_sub
    return tp, fp, fn, bias, mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="data/ground_truth_v2.csv")
    parser.add_argument("--sub", default="output/submission_skip2.csv")
    parser.add_argument("--timed", default=None,
                        help="output of run_pipeline.py --timed_log; required for method 3")
    parser.add_argument("--time_tolerance", type=float, default=3.0)
    args = parser.parse_args()

    gt_full = load_gt(args.gt)
    gt_keys = [(cls, action) for _, cls, action in gt_full]
    sub_keys = load_submission(args.sub)

    print(f"GT events: {len(gt_keys)}  Submission events: {len(sub_keys)}\n")

    print("=== Method 1: count-only (Counter, order-agnostic) ===")
    tp, fp, fn, details = score_count(gt_keys, sub_keys)
    print_score("count", tp, fp, fn)
    for cls, action, g, s in sorted(details, key=lambda x: -abs(x[2] - x[3]))[:10]:
        diff = s - g
        print(f"    {'+' if diff>0 else '-'}{abs(diff)}  {cls:<50} {action}  GT={g} Sub={s}")
    print()

    print("=== Method 2: global order (class-agnostic LCS) ===")
    tp, fp, fn = score_order(gt_keys, sub_keys)
    print_score("order", tp, fp, fn)
    print()

    if args.timed:
        print("=== Method 3: time-value matching (delay-bias corrected) ===")
        sub_timed = load_timed(args.timed)
        tp, fp, fn, bias, mismatches = score_time(gt_full, sub_timed, args.time_tolerance)
        print(f"  estimated detection delay (median): {bias:+.2f}s")
        print_score(f"time ±{args.time_tolerance}s", tp, fp, fn)
        if mismatches:
            print(f"\n  mismatched beyond tolerance ({len(mismatches)}):")
            for cls, action, g, s, d in sorted(mismatches, key=lambda x: -x[4])[:10]:
                print(f"    {cls:<50} {action:<8} GT={g:.1f}s Sub={s:.1f}s diff={d:.1f}s")
    else:
        print("(skipping Method 3 -- pass --timed <path> from run_pipeline.py --timed_log)")


if __name__ == "__main__":
    main()
