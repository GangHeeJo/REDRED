"""
per_cam_log CSV를 분석해 CLASS_CAM_WHITELIST 후보 추천.

Usage:
    python tools/analyze_per_cam.py \
        --per_cam output/per_cam_kd_clean.csv \
        --names   data/names.txt \
        [--focus  cholula hunts_sauce dr_pepper hersheys_cocoa quaker_big_chewy]

출력:
  1. 지정 클래스별 카메라 검출 빈도 (전체 영상 기준)
  2. 관심 구간(frame range) 안에서 카메라별 검출 현황
  3. 추천 whitelist (검출률 >= threshold 카메라만)
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_names(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_per_cam(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame": int(row["frame_idx"]),
                "cam":   int(row["cam_id"]),
                "cls":   int(row["class_id"]),
                "name":  row["class_name"],
                "count": int(row["count"]),
            })
    return rows


def analyze(rows, class_names, focus_names, min_rate=0.05):
    # 전체 프레임 수 추정
    all_frames = sorted(set(r["frame"] for r in rows))
    total_frames = len(all_frames)
    n_cams = 5

    # class_id 매핑
    name2id = {n: i for i, n in enumerate(class_names)}
    focus_ids = set()
    for n in focus_names:
        if n in name2id:
            focus_ids.add(name2id[n])
        else:
            # partial match
            for cname, cid in name2id.items():
                if n.lower() in cname.lower():
                    focus_ids.add(cid)

    if not focus_ids:
        focus_ids = set(r["cls"] for r in rows)

    # per-class, per-cam detection frame count
    det_frames = defaultdict(lambda: defaultdict(set))  # cls → cam → set of frames
    for r in rows:
        if r["cls"] in focus_ids:
            det_frames[r["cls"]][r["cam"]].add(r["frame"])

    print(f"\n{'='*70}")
    print(f"총 프레임(검출 발생): {total_frames}  카메라: {n_cams}대")
    print(f"분석 대상 클래스: {len(focus_ids)}개")
    print(f"{'='*70}\n")

    whitelist_suggestions = {}

    for cls_id in sorted(focus_ids):
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        cam_data = det_frames[cls_id]

        if not cam_data:
            print(f"[{cls_id:2d}] {name}: 검출 없음")
            continue

        print(f"[{cls_id:2d}] {name}")

        cam_rates = {}
        for cam in range(n_cams):
            n_det = len(cam_data.get(cam, set()))
            rate  = n_det / total_frames
            cam_rates[cam] = rate
            bar = "█" * int(rate * 200)
            print(f"  cam{cam}: {n_det:5d}프레임 ({rate*100:5.1f}%)  {bar}")

        # 추천: 검출률 >= min_rate인 카메라만 whitelist
        whitelist = [c for c, r in cam_rates.items() if r >= min_rate]
        # 검출하는 카메라가 아예 없거나 전부면 whitelist 불필요
        if 0 < len(whitelist) < n_cams:
            whitelist_suggestions[cls_id] = sorted(whitelist)
            print(f"  → 추천 whitelist(검출률≥{min_rate*100:.0f}%): cam{whitelist}")
        else:
            print(f"  → whitelist 불필요 (모든 카메라 or 검출 없음)")
        print()

    print("=" * 70)
    print("CLASS_CAM_WHITELIST 제안 (코드에 붙여넣기):")
    print("CLASS_CAM_WHITELIST: Dict[int, List[int]] = {")
    for cls_id, cams in sorted(whitelist_suggestions.items()):
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        print(f"    {cls_id}: {cams},   # {name}")
    print("}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_cam", default="output/per_cam_kd_clean.csv")
    parser.add_argument("--names",   default="data/names.txt")
    parser.add_argument("--focus",   nargs="*", default=[
        "cholula", "hunts_sauce", "dr_pepper",
        "hersheys_cocoa", "hersheys_bar", "quaker_big_chewy",
        "campbells_chicken_noodle_soup", "bulls_eye", "redbull",
        "palmolive", "crystal_hot_sauce",
    ])
    parser.add_argument("--min_rate", type=float, default=0.05,
                        help="whitelist에 포함할 최소 검출률 (기본 5%%)")
    args = parser.parse_args()

    class_names = load_names(args.names)
    rows = load_per_cam(args.per_cam)
    analyze(rows, class_names, args.focus or [], args.min_rate)


if __name__ == "__main__":
    main()
