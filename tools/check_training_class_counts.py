"""
Count how many training label instances exist per class, to check whether
classes with poor live-detection recall (e.g. bumblebee_albacore, dove_white,
dove_pink, nabisco_nilla_wafers) are simply underrepresented in the YOLO
training set.

Run on the server (no GPU needed, just reads label .txt files):
    PYTHONPATH=~/yolov7 python tools/check_training_class_counts.py \
        --train_txt ~/yolov7/data/train.txt \
        --names data/names.txt

Assumes standard YOLO layout: each image path has a sibling label .txt found
by replacing "/images/" with "/labels/" and the image extension with ".txt".
Falls back to same-directory same-basename .txt if that path doesn't exist.
"""

import argparse
import os
import sys
import time
from collections import Counter


def find_label_path(img_path: str) -> str:
    base, _ = os.path.splitext(img_path)
    candidate = img_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    candidate = os.path.splitext(candidate)[0] + ".txt"
    if os.path.exists(candidate):
        return candidate
    fallback = base + ".txt"
    if os.path.exists(fallback):
        return fallback
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--names", required=True)
    args = parser.parse_args()

    with open(args.names, encoding="utf-8") as f:
        class_names = [l.strip() for l in f if l.strip()]

    with open(args.train_txt, encoding="utf-8") as f:
        img_paths = [l.strip() for l in f if l.strip()]

    print(f"Found {len(img_paths)} image paths in {args.train_txt}, scanning labels...")

    instance_counts = Counter()   # class_id -> total bbox instances
    image_counts = Counter()      # class_id -> number of images containing it
    missing_labels = 0
    t0 = time.time()

    for i, img_path in enumerate(img_paths, start=1):
        if i % 1000 == 0 or i == len(img_paths):
            elapsed = time.time() - t0
            print(f"  ...{i}/{len(img_paths)} images scanned ({elapsed:.1f}s elapsed)", flush=True)
        label_path = find_label_path(img_path)
        if not label_path:
            missing_labels += 1
            continue
        seen_classes_this_image = set()
        with open(label_path, encoding="utf-8") as lf:
            for line in lf:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                instance_counts[cls_id] += 1
                seen_classes_this_image.add(cls_id)
        for cls_id in seen_classes_this_image:
            image_counts[cls_id] += 1

    print(f"Total training images listed: {len(img_paths)}")
    print(f"Images with no findable label file: {missing_labels}\n")

    print(f"{'class_id':>8}  {'class_name':<55} {'images':>8} {'instances':>10}")
    rows = []
    for cls_id, name in enumerate(class_names):
        rows.append((cls_id, name, image_counts.get(cls_id, 0), instance_counts.get(cls_id, 0)))

    rows.sort(key=lambda r: r[2])  # ascending by image count -> worst first
    for cls_id, name, n_img, n_inst in rows:
        print(f"{cls_id:>8}  {name:<55} {n_img:>8} {n_inst:>10}")

    print("\n=== Specifically flagged classes ===")
    flagged = ["bumblebee_albacore", "dove_white", "dove_pink", "nabisco_nilla_wafers",
               "haribo_gold_bears_gummi_candy"]
    name_to_row = {r[1]: r for r in rows}
    for name in flagged:
        if name in name_to_row:
            cls_id, _, n_img, n_inst = name_to_row[name]
            pct_rank = sum(1 for r in rows if r[2] <= n_img) / len(rows) * 100
            print(f"  {name:<55} images={n_img:<6} instances={n_inst:<6} "
                  f"(percentile from bottom: {pct_rank:.0f}%)")
        else:
            print(f"  {name}: NOT FOUND in names.txt")


if __name__ == "__main__":
    main()
