"""
Detection quality analysis at ground truth event frames.

Inputs:
  data/ground_truth.csv          - 105 verified events with frame numbers
  output/debug_frame_counts.csv  - per-frame fused detection counts

Reports:
  1. Per-event: was the right class detected near the GT frame?
  2. Confusion: what else was detected at GT frames? (similar-class errors)
  3. Double detection: classes with count >= 2 (same item detected twice)
  4. Problem class summary

Usage:
  python tools/analyze_detections.py
  python tools/analyze_detections.py --window 100 --debug output/debug_frame_counts.csv
"""

import csv
import argparse
from collections import defaultdict

WINDOW = 60   # real frames around GT event to check


def load_ground_truth(path):
    events = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("frame"):
                events.append({
                    "event_num":  int(row["event_num"]),
                    "section":    int(row["section"]),
                    "class_name": row["class_name"],
                    "action":     row["action"],
                    "frame":      int(row["frame"]),
                })
    return events


def load_debug_counts(path):
    """Returns {frame_idx: {class_name: count}}"""
    data = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            f_idx = int(row["frame_idx"])
            cls   = row["class_name"]
            cnt   = int(row["count"])
            data[f_idx][cls] = cnt
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",     default="data/ground_truth.csv")
    parser.add_argument("--debug",  default="output/debug_frame_counts.csv")
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    gt_events  = load_ground_truth(args.gt)
    frame_data = load_debug_counts(args.debug)

    print(f"GT events with frames: {len(gt_events)}")
    print(f"Debug frames loaded:   {len(frame_data)}")
    print(f"Search window:         ±{args.window} real frames\n")

    # ── 1. Per-event detection check ───────────────────────────────
    print("=" * 100)
    print("GT 이벤트별 감지 현황")
    print("=" * 100)
    print(f"{'#':>3} {'S':>2} {'클래스':<52} {'액션':<5} {'GT':>7} {'감지프레임':>10} {'count':>5}  판정")
    print("-" * 100)

    class_stats   = defaultdict(lambda: {"ok": 0, "miss": 0, "double": 0})
    confusion_log = []  # (gt_frame, target_cls, other_cls, other_cnt)

    for ev in gt_events:
        cls      = ev["class_name"]
        gt_frame = ev["frame"]
        w        = args.window

        best_cnt   = 0
        best_frame = None
        all_dets_in_window = defaultdict(int)  # other class detections in window

        for f in range(gt_frame - w, gt_frame + w + 1):
            if f not in frame_data:
                continue
            dets = frame_data[f]
            if cls in dets:
                if dets[cls] > best_cnt:
                    best_cnt   = dets[cls]
                    best_frame = f
            for other_cls, cnt in dets.items():
                if other_cls != cls:
                    all_dets_in_window[other_cls] = max(all_dets_in_window[other_cls], cnt)

        if best_cnt == 0:
            verdict = "누락  ❌"
            class_stats[cls]["miss"] += 1
        elif best_cnt == 1:
            verdict = "정상  ✓"
            class_stats[cls]["ok"] += 1
        else:
            verdict = f"중복({best_cnt}) ⚠"
            class_stats[cls]["double"] += 1

        bf_str = str(best_frame) if best_frame is not None else "-"
        print(f"{ev['event_num']:>3} {ev['section']:>2} {cls:<52} {ev['action']:<5} "
              f"{gt_frame:>7} {bf_str:>10} {best_cnt:>5}  {verdict}")

        # Log top confused class at this event frame
        if all_dets_in_window:
            top_other = sorted(all_dets_in_window.items(), key=lambda x: -x[1])[:2]
            for other_cls, cnt in top_other:
                confusion_log.append((gt_frame, cls, other_cls, cnt, ev["action"]))

    # ── 2. Double detection summary ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Double Detection (count >= 2) 빈도 상위 클래스")
    print("=" * 60)
    double_freq = defaultdict(int)
    for dets in frame_data.values():
        for cls, cnt in dets.items():
            if cnt >= 2:
                double_freq[cls] += 1
    for cls, freq in sorted(double_freq.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * min(freq // 10, 40)
        print(f"  {cls:<52} {freq:>5}프레임  {bar}")

    # ── 3. Confusion at GT frames ───────────────────────────────────
    print("\n" + "=" * 60)
    print("GT 이벤트 근처 함께 감지된 클래스 (혼동 후보)")
    print("=" * 60)
    pair_freq = defaultdict(int)
    for _, target, other, _, _ in confusion_log:
        pair_freq[(target, other)] += 1
    for (t, o), freq in sorted(pair_freq.items(), key=lambda x: -x[1])[:15]:
        print(f"  {t:<40} ← 혼동 → {o:<40} {freq:>3}건")

    # ── 4. Problem class summary ────────────────────────────────────
    print("\n" + "=" * 60)
    print("클래스별 문제 요약")
    print("=" * 60)
    print(f"  {'클래스':<52} {'정상':>5} {'누락':>5} {'중복':>5}")
    print("  " + "-" * 72)
    for cls, s in sorted(class_stats.items(), key=lambda x: -(x[1]["miss"] + x[1]["double"])):
        if s["miss"] > 0 or s["double"] > 0:
            print(f"  {cls:<52} {s['ok']:>5} {s['miss']:>5} {s['double']:>5}")

    # ── 5. Overall stats ────────────────────────────────────────────
    total_ok     = sum(s["ok"]     for s in class_stats.values())
    total_miss   = sum(s["miss"]   for s in class_stats.values())
    total_double = sum(s["double"] for s in class_stats.values())
    total        = total_ok + total_miss + total_double
    print(f"\n총 {total}개 GT 이벤트 (프레임 있는 것만)")
    print(f"  정상: {total_ok} ({100*total_ok/total:.1f}%)")
    print(f"  누락: {total_miss} ({100*total_miss/total:.1f}%)")
    print(f"  중복: {total_double} ({100*total_double/total:.1f}%)")


if __name__ == "__main__":
    main()
