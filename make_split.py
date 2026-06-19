"""
Split image list into train/val.

Reads all image paths from the dataset (or from make_list_dir.py output),
shuffles, then writes train.txt and val.txt.

Usage:
    python make_split.py \
        --img_dir ~/Dataset/1.competition_trainset \
        --out_dir ~/yolov7/data \
        --val_ratio 0.1 \
        --seed 42

    # Or from an existing list file:
    python make_split.py \
        --list_file ~/yolov7/data/target.txt \
        --out_dir   ~/yolov7/data \
        --val_ratio 0.1
"""

import argparse
import random
import os
import glob


def collect_from_dir(img_dir):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
    return sorted(paths)


def collect_from_list(list_file):
    with open(list_file) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",   default=None)
    parser.add_argument("--list_file", default=None)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    if args.list_file:
        paths = collect_from_list(args.list_file)
    elif args.img_dir:
        paths = collect_from_dir(args.img_dir)
    else:
        raise ValueError("Provide --img_dir or --list_file")

    random.seed(args.seed)
    random.shuffle(paths)

    n_val = max(1, int(len(paths) * args.val_ratio))
    val_paths   = paths[:n_val]
    train_paths = paths[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_paths))
    with open(os.path.join(args.out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_paths))

    print(f"Total: {len(paths)}  |  Train: {len(train_paths)}  |  Val: {len(val_paths)}")
    print(f"Wrote → {args.out_dir}/train.txt, val.txt")


if __name__ == "__main__":
    main()
