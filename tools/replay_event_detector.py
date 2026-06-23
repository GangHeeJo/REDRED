"""
Replay the real EventDetector against the recorded debug_frame_counts.csv
(raw fused counts before smoothing), printing every internal state
transition for a chosen set of classes. Pure stdlib + event_detector.py,
no GPU/model needed -- runs entirely on the already-pulled debug log.

Usage:
    python tools/replay_event_detector.py [class_name ...]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from event_detector import EventDetector, WINDOW_SIZE, MAX_DELTA, CONFIRM_FRAMES, MAX_INVENTORY

DEBUG_LOG = "output/debug_frame_counts.csv"
NAMES_PATH = "data/names.txt"


def load_names():
    with open(NAMES_PATH, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_debug_log(path):
    """frame_idx -> {class_id: count}, plus sorted list of all frame_idx seen."""
    per_frame = defaultdict(dict)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            per_frame[int(r["frame_idx"])][int(r["class_id"])] = int(r["count"])
    return per_frame


def main():
    full_replay = "--full" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--full"]
    target_names = args or ["coffee_mate_french_vanilla", "jif_creamy_peanut_butter",
                             "pop_tararts_strawberry", "hunts_sauce"]

    class_names = load_names()
    name_to_id = {n: i for i, n in enumerate(class_names)}
    target_ids = set(range(len(class_names))) if full_replay else {name_to_id[n] for n in target_names if n in name_to_id}
    for n in target_names:
        if n not in name_to_id:
            print(f"WARNING: class '{n}' not found in names.txt")

    per_frame = load_debug_log(DEBUG_LOG)

    import cv2
    cap = cv2.VideoCapture(str(Path(__file__).parent.parent / "cam0_Sample1.mp4"))
    n_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    skip = 2
    all_frame_idx = list(range(0, n_raw_frames, skip))
    print(f"Replaying ALL {len(all_frame_idx)} processed frames (0..{n_raw_frames-1} step {skip}), "
          f"{len(per_frame)} of which have >=1 nonzero detection in the debug log\n")

    # Reconstruct initial_counts from frames 0..29 (mirrors estimate_initial_inventory)
    init_window = [f for f in all_frame_idx if f < 30]
    init_hist = defaultdict(list)
    for f in init_window:
        for cls_id, cnt in per_frame[f].items():
            init_hist[cls_id].append(cnt)
    import statistics
    initial_counts = {
        cls_id: int(statistics.median(vals))
        for cls_id, vals in init_hist.items()
        if int(statistics.median(vals)) > 0
    }

    print(f"Params: WINDOW_SIZE={WINDOW_SIZE} MAX_DELTA={MAX_DELTA} "
          f"CONFIRM_FRAMES={CONFIRM_FRAMES} MAX_INVENTORY={MAX_INVENTORY}")
    print(f"Initial counts for targets: "
          f"{ {class_names[c]: v for c, v in initial_counts.items() if c in target_ids} }\n")

    detector = EventDetector(class_names, initial_counts=initial_counts)

    last_state = {cid: None for cid in target_ids}
    last_committed = {cid: None for cid in target_ids}
    last_candidate = {cid: None for cid in target_ids}

    for frame_idx in all_frame_idx:
        frame_counts = per_frame[frame_idx]
        dets = [{"class_id": cid, "confidence": 1.0, "bbox": []}
                for cid, cnt in frame_counts.items() for _ in range(cnt)]

        events = detector.update(dets)

        for ev in events:
            if ev.class_id in target_ids:
                print(f"[FRAME {frame_idx}] *** EVENT FIRED *** {ev.class_name} "
                      f"{ev.action} ({ev.before}->{ev.after})")

        for cid in target_ids:
            state = detector._sm_state[cid]
            committed = detector._committed[cid]
            candidate = detector._candidate.get(cid)
            median = detector._median(cid)
            raw = frame_counts.get(cid, 0)

            changed = (state != last_state[cid] or committed != last_committed[cid]
                       or candidate != last_candidate[cid])
            if changed:
                print(f"[FRAME {frame_idx}] {class_names[cid]:<45} raw={raw} median={median} "
                      f"state={state:<10} committed={committed} candidate={candidate}")
            last_state[cid] = state
            last_committed[cid] = committed
            last_candidate[cid] = candidate

    if full_replay:
        from csv_generator import load_prices, events_to_csv
        prices = load_prices("data/prices.csv")
        out_path = "output/submission_LOCAL_REPLAY.csv"
        events_to_csv(
            events=detector.all_events,
            prices=prices,
            out_path=out_path,
            initial_inventory=initial_counts,
            include_action=True,
            total_mode="per_class",
        )
        print(f"\nTotal events fired by detector: {len(detector.all_events)}")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
