"""
Method 2(order/LCS)가 정확히 어느 이벤트 쌍에서 깨지는지 추적.
score_order()는 tp/fp/fn 개수만 반환하는데, 여기선 실제 LCS 정렬 결과를
GT/Sub 원문 시퀀스와 함께 출력해서 어느 클래스 이벤트가 순서를 깨는지 확인.

Usage:
    python tools/diagnose_order_mismatch.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_xxx.csv \
        --timed output/timed_native.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score_methods import load_gt, load_submission, load_timed, ACTION_MAP


def lcs_with_trace(gt_seq, sub_seq):
    n, m = len(gt_seq), len(sub_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_seq[i - 1] == sub_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # traceback
    i, j = n, m
    matched = []  # (gt_idx, sub_idx)
    while i > 0 and j > 0:
        if gt_seq[i - 1] == sub_seq[j - 1]:
            matched.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    matched.reverse()
    matched_gt = {g for g, s in matched}
    matched_sub = {s for g, s in matched}
    return matched, matched_gt, matched_sub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="data/ground_truth_v2.csv")
    p.add_argument("--sub", required=True)
    p.add_argument("--timed", required=True)
    args = p.parse_args()

    gt_full = load_gt(args.gt)  # (time, cls, action) sorted by time
    gt_keys = [(cls, action) for _, cls, action in gt_full]
    sub_keys = load_submission(args.sub)  # (cls, action) in firing order
    sub_timed = load_timed(args.timed)  # (time, cls, action) sorted by time -- for reference only

    # sub_keys is in firing order already (== chronological per csv_generator).
    # build a time lookup for sub events by matching order to sub_timed (should align 1:1 in count)
    sub_times = [t for t, _, _ in sub_timed]

    matched, matched_gt, matched_sub = lcs_with_trace(gt_keys, sub_keys)

    print(f"GT={len(gt_keys)}  Sub={len(sub_keys)}  Matched(LCS)={len(matched)}\n")

    print("=== GT에서 매칭 안 된 이벤트 (order 기준 FN) ===")
    for idx, (t, cls, action) in enumerate(gt_full):
        if idx not in matched_gt:
            print(f"  GT#{idx:3d}  t={t:6.1f}s  {cls:<45} {action}")

    print("\n=== Sub에서 매칭 안 된 이벤트 (order 기준 FP) ===")
    for idx, (cls, action) in enumerate(sub_keys):
        if idx not in matched_sub:
            t = sub_times[idx] if idx < len(sub_times) else float("nan")
            print(f"  Sub#{idx:3d}  t={t:6.1f}s  {cls:<45} {action}")

    print("\n=== 매칭된 이벤트 중 GT/Sub 시각 근접해서 순서 흔들릴 뻔한 것들 (참고용, 매칭 자체는 성공) ===")
    for k, (gi, si) in enumerate(matched):
        if k == 0:
            continue
        prev_gi, prev_si = matched[k - 1]
        gt_gap = gt_full[gi][0] - gt_full[prev_gi][0]
        if gt_gap < 3.0:
            print(f"  GT#{prev_gi}({gt_full[prev_gi][1]},{gt_full[prev_gi][0]:.1f}s) -> "
                  f"GT#{gi}({gt_full[gi][1]},{gt_full[gi][0]:.1f}s)  gap={gt_gap:.1f}s")


if __name__ == "__main__":
    main()
