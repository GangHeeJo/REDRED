"""
기존 YOLOv7 학습 데이터(YOLO format) → RF-DETR COCO 포맷 변환

RF-DETR 기대 구조:
    data/coco_rfdetr/
        train/
            _annotations.coco.json
            img1.jpg ...
        valid/
            _annotations.coco.json
            img1.jpg ...

Usage:
    python tools/yolo_to_coco.py \
        --train_txt ~/yolov7/data/train.txt \
        --names     data/names.txt \
        --out_dir   data/coco_rfdetr
"""

import argparse
import json
import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_txt", default=os.path.expanduser("~/yolov7/data/train.txt"))
    p.add_argument("--names",     default="data/names.txt")
    p.add_argument("--out_dir",   default="data/coco_rfdetr")
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--symlink",   action="store_true")
    return p.parse_args()


def load_names(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def yolo_to_coco(image_paths, names, split, out_dir, symlink=False):
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i, "name": n, "supercategory": "product"}
                  for i, n in enumerate(names)]
    images, annotations = [], []
    ann_id = 1

    for img_path in tqdm(image_paths, desc=split):
        img_path = Path(img_path)
        label_path = img_path.with_suffix(".txt")
        label_path_alt = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        lp = label_path if label_path.exists() else label_path_alt
        if not lp.exists():
            continue

        try:
            img = Image.open(img_path)
            W, H = img.size
        except Exception:
            continue

        dst = split_dir / img_path.name
        if not dst.exists():
            if symlink:
                dst.symlink_to(img_path.resolve())
            else:
                shutil.copy2(img_path, dst)

        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": img_path.name, "width": W, "height": H})

        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x = (cx - bw / 2) * W
                y = (cy - bh / 2) * H
                w, h = bw * W, bh * H
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cls_id,
                    "bbox": [round(x,2), round(y,2), round(w,2), round(h,2)],
                    "area": round(w * h, 2), "iscrowd": 0,
                })
                ann_id += 1

    out_json = split_dir / "_annotations.coco.json"
    with open(out_json, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    print(f"[{split}] {len(images)} images, {len(annotations)} annotations → {out_json}")


def main():
    args = parse_args()
    names = load_names(args.names)

    with open(args.train_txt) as f:
        all_paths = [l.strip() for l in f if l.strip()]

    random.shuffle(all_paths)
    n_val = max(1, int(len(all_paths) * args.val_ratio))
    val_paths   = all_paths[:n_val]
    train_paths = all_paths[n_val:]

    yolo_to_coco(train_paths, names, "train", args.out_dir, args.symlink)
    yolo_to_coco(val_paths,   names, "valid", args.out_dir, args.symlink)  # rfdetr는 'valid'
    print(f"\nDone. Dataset: {args.out_dir}")


if __name__ == "__main__":
    main()
