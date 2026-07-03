"""
per_cam_native.csv 분석 -- 문제 클래스들의 카메라별 검출률 실측.
추측 대신 데이터로 whitelist/quorum/presence_threshold 재산출하기 위함.

Usage:
    python tools/analyze_per_cam_native.py \
        --per_cam output/per_cam_native.csv \
        --names data/names.txt \
        --focus 0 6 8 23 43 45 47 48 51
"""
import argparse
import csv
from collections import defaultdict


def load_names(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_cam", default="output/per_cam_native.csv")
    p.add_argument("--names", default="data/names.txt")
    p.add_argument("--focus", nargs="*", type=int, default=[0, 6, 8, 23, 43, 45, 47, 48, 51])
    args = p.parse_args()

    names = load_names(args.names)

    det_frames = defaultdict(lambda: defaultdict(set))  # cls -> cam -> set of frame_idx
    all_frames = set()
    with open(args.per_cam, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame_idx"])
            all_frames.add(frame)
            cls = int(row["class_id"])
            cam = int(row["cam_id"])
            if cls in args.focus:
                det_frames[cls][cam].add(frame)

    total_frames = len(all_frames)
    print(f"총 처리 프레임 수: {total_frames}\n")

    for cls_id in args.focus:
        name = names[cls_id] if cls_id < len(names) else f"cls_{cls_id}"
        cam_data = det_frames[cls_id]
        print(f"[{cls_id:2d}] {name}")
        if not cam_data:
            print("  검출 없음 (전 카메라)\n")
            continue
        for cam in range(5):
            n = len(cam_data.get(cam, set()))
            rate = n / total_frames if total_frames else 0
            bar = "#" * int(rate * 100)
            print(f"  cam{cam}: {n:5d}프레임 ({rate*100:5.1f}%)  {bar}")
        print()


if __name__ == "__main__":
    main()
