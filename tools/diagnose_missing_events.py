"""
For each ground-truth event with a known frame number, inspect the raw
per-frame fused-count debug log (from run_pipeline.py --debug_log) in a
window around that frame to tell apart two failure modes:

  - PERCEPTION FAILURE: raw count never reflects the expected post-event
    inventory level nearby -> the detector genuinely never saw it.
  - LOGIC-BLOCKED: raw count DID reach the expected level for a while near
    the event time, but no event was fired in the submission -> the
    EventDetector's state machine (CONFIRM_FRAMES / inventory constraints)
    swallowed a real detection.

Usage:
    python tools/diagnose_missing_events.py [debug_log_csv] [window_frames]
"""

import csv
import sys
from collections import defaultdict

GT_PATH = "data/ground_truth.csv"
WINDOW = int(sys.argv[2]) if len(sys.argv) > 2 else 150  # +/- frames (skip=2 -> ~10s)


def load_ground_truth(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_debug_log(path):
    """class_name -> list of (frame_idx, count), sorted by frame_idx."""
    per_class = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            per_class[r["class_name"]].append((int(r["frame_idx"]), int(r["count"])))
    for cls in per_class:
        per_class[cls].sort()
    return per_class


def counts_in_window(series, center_frame, window):
    return [(f, c) for f, c in series if abs(f - center_frame) <= window]


def main():
    debug_path = sys.argv[1] if len(sys.argv) > 1 else "output/debug_frame_counts.csv"
    gt = load_ground_truth(GT_PATH)
    debug = load_debug_log(debug_path)

    gt_with_frame = [r for r in gt if r["frame"]]
    print(f"Ground truth events with known frame: {len(gt_with_frame)} / {len(gt)}")
    print(f"Window: +/- {WINDOW} frames\n")

    perception_failure = []
    logic_blocked = []
    unclear = []

    for r in gt_with_frame:
        cls = r["class_name"]
        frame = int(r["frame"])
        action = r["action"]
        series = debug.get(cls, [])
        nearby = counts_in_window(series, frame, WINDOW)

        if action == "return":
            # expect count to rise to >=1 around this frame
            saw_present = any(c >= 1 for _, c in nearby)
        else:  # purchase
            # expect count to drop to 0 around this frame (i.e. some 0 nearby
            # following prior presence) -- weaker check, just look for any
            # detection at all nearby as a sanity signal
            saw_present = any(c >= 1 for _, c in nearby)

        max_count = max((c for _, c in nearby), default=0)
        n_frames_present = sum(1 for _, c in nearby if c >= 1)

        entry = {
            "event_num": r["event_num"], "section": r["section"], "class_name": cls,
            "action": action, "frame": frame, "max_count": max_count,
            "n_frames_present": n_frames_present, "n_samples": len(nearby),
        }

        if action == "return":
            if max_count == 0:
                perception_failure.append(entry)
            elif n_frames_present < 30:  # CONFIRM_FRAMES threshold in event_detector.py
                logic_blocked.append(entry)
            else:
                unclear.append(entry)  # detected long enough but still missing -> investigate (inventory constraint?)

    print(f"=== RETURN events: raw count NEVER appeared nearby (perception failure) : {len(perception_failure)} ===")
    for e in perception_failure:
        print(f"  ev{e['event_num']:>3} sec{e['section']} {e['class_name']:<50} frame={e['frame']} max_count={e['max_count']} present_frames={e['n_frames_present']}/{e['n_samples']}")

    print(f"\n=== RETURN events: raw count appeared but briefly (<30 frames, likely too short to confirm) : {len(logic_blocked)} ===")
    for e in logic_blocked:
        print(f"  ev{e['event_num']:>3} sec{e['section']} {e['class_name']:<50} frame={e['frame']} max_count={e['max_count']} present_frames={e['n_frames_present']}/{e['n_samples']}")

    print(f"\n=== RETURN events: raw count present >=30 frames nearby but still missing from submission (constraint/logic bug?) : {len(unclear)} ===")
    for e in unclear:
        print(f"  ev{e['event_num']:>3} sec{e['section']} {e['class_name']:<50} frame={e['frame']} max_count={e['max_count']} present_frames={e['n_frames_present']}/{e['n_samples']}")


if __name__ == "__main__":
    main()
