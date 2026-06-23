"""
Compare a generated submission CSV against data/ground_truth.csv.

Ground truth and submission don't share a common key (submission has no
frame/time column), so events are aligned as ordered (class_name, action)
sequences using difflib's longest-matching-blocks algorithm. This finds the
best alignment under the assumption both lists are in roughly chronological
order, then reports what ground truth events were missed and what
submission events don't correspond to any ground truth event.

Usage:
    python tools/compare_to_ground_truth.py [submission_csv]
    (default: output/submission_skip2.csv)
"""

import csv
import sys
import difflib
from collections import Counter

GT_PATH = "data/ground_truth.csv"
ACTION_KO2EN = {"구매": "purchase", "반환": "return"}


def load_ground_truth(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "event_num": int(r["event_num"]),
                "section": r["section"],
                "class_name": r["class_name"],
                "action": r["action"],
                "frame": r["frame"] or None,
                "time_sec": r["time_sec"] or None,
            })
    return rows


def load_submission(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            action_ko = r["구매/반환 여부"].strip()
            rows.append({
                "event_num": r["이벤트 번호"],
                "class_name": r["품목명"],
                "action": ACTION_KO2EN.get(action_ko, action_ko),
            })
    return rows


def align(gt, sub):
    gt_keys = [(g["class_name"], g["action"]) for g in gt]
    sub_keys = [(s["class_name"], s["action"]) for s in sub]

    sm = difflib.SequenceMatcher(a=gt_keys, b=sub_keys, autojunk=False)
    matched_gt_idx = set()
    matched_sub_idx = set()
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            matched_gt_idx.add(block.a + k)
            matched_sub_idx.add(block.b + k)

    missing = [gt[i] for i in range(len(gt)) if i not in matched_gt_idx]
    extra = [sub[i] for i in range(len(sub)) if i not in matched_sub_idx]
    return matched_gt_idx, missing, extra


def main():
    sub_path = sys.argv[1] if len(sys.argv) > 1 else "output/submission_skip2.csv"

    gt = load_ground_truth(GT_PATH)
    sub = load_submission(sub_path)

    matched_idx, missing, extra = align(gt, sub)

    print(f"Ground truth events : {len(gt)}")
    print(f"Submission events   : {len(sub)}")
    print(f"Matched             : {len(matched_idx)}")
    print(f"Missing (false neg) : {len(missing)}")
    print(f"Extra   (false pos) : {len(extra)}")
    print()

    print("=== Missing events (in ground truth, not in submission) ===")
    for m in missing:
        t = f"t={m['time_sec']}s" if m["time_sec"] else "t=?"
        print(f"  event {m['event_num']:>3} (section {m['section']}) "
              f"{m['class_name']:<55} {m['action']:<8} {t}")

    print()
    print("=== Extra events (in submission, no matching ground truth) ===")
    for e in extra:
        print(f"  event {e['event_num']:>3} {e['class_name']:<55} {e['action']}")

    print()
    print("=== Missing events by class (counts) ===")
    for cls, n in Counter(m["class_name"] for m in missing).most_common():
        print(f"  {cls:<55} {n}")

    print()
    print("=== Extra events by class (counts) ===")
    for cls, n in Counter(e["class_name"] for e in extra).most_common():
        print(f"  {cls:<55} {n}")


if __name__ == "__main__":
    main()
