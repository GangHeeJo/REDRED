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
    p.add_argument("--oversample_weak", type=int, default=0,
                    help="WEAK_CLASS_IDS 포함 이미지를 이 배수만큼 train에 복제 (0=비활성)")
    return p.parse_args()


def load_names(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


# tools/sam2_video_label.py의 WEAK_CLASS_IDS와 동일 (2026-07-02: 파이프라인
# 튜닝으로 못 잡은 미탐지 클래스 -- 학습 노출 빈도를 인위적으로 높여서 보강 시도)
WEAK_CLASS_IDS = {0, 8, 43, 45, 48}  # aunt_jemima, hunts_sauce, campbells,
                                     # chewy_dips_chocolate_chip, cheerios

# 2026-07-02 추가: 미탐지는 아니지만(정상 검출됨) 타이밍 오차가 큰 클래스
# (score_methods.py Method 3, ±3초 초과 중 5초 이상). 이미 잘 잡히니 SAM2
# 캡처 문턱은 안 낮춤(sam2_video_label.py는 안 건드림) -- 오버샘플링만으로
# confidence를 더 뾰족하게(빨리 안정되게) 만들 수 있는지 시도.
TIMING_ISSUE_CLASS_IDS = {2, 22, 42, 46, 50}  # bumblebee_albacore,
    # haribo_gold_bears_gummi_candy, pepperidge_farm_milano_cookies_double_chocolate,
    # chewy_dips_peanut_butter, lindt_excellence_cocoa_dark_chocolate

REINFORCE_CLASS_IDS = WEAK_CLASS_IDS | TIMING_ISSUE_CLASS_IDS


def label_path_for(img_path):
    img_path = Path(img_path)
    p1 = img_path.with_suffix(".txt")
    p2 = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    return p1 if p1.exists() else p2


def contains_reinforce_class(img_path):
    lp = label_path_for(img_path)
    if not lp.exists():
        return False
    with open(lp) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) in REINFORCE_CLASS_IDS:
                return True
    return False


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

    if args.oversample_weak > 0:
        weak_paths = [p for p in tqdm(train_paths, desc="scanning for reinforce classes")
                      if contains_reinforce_class(p)]
        extra = weak_paths * (args.oversample_weak - 1)
        train_paths = train_paths + extra
        random.shuffle(train_paths)
        print(f"Oversampled {len(weak_paths)} reinforce-class images x{args.oversample_weak} "
              f"(+{len(extra)} duplicated entries, train now {len(train_paths)} total)")

    yolo_to_coco(train_paths, names, "train", args.out_dir, args.symlink)
    yolo_to_coco(val_paths,   names, "valid", args.out_dir, args.symlink)  # rfdetr는 'valid'
    print(f"\nDone. Dataset: {args.out_dir}")


if __name__ == "__main__":
    main()
