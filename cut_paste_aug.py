"""
Cut & Paste Augmentation for unmanned vending machine dataset.

Usage:
    python cut_paste_aug.py \
        --seg_dir ~/Dataset/2.backsub_images_100 \
        --bg_dir  ~/Dataset/3.Background_Images \
        --out_dir ~/Dataset/augmented \
        --num_images 5000 \
        --max_objects 4

Output layout:
    out_dir/
        images/aug_00000.jpg ...
        labels/aug_00000.txt ...   (YOLO format: class cx cy w h)
"""

import argparse
import random
import os
import glob
import cv2
import numpy as np


def load_seg_images(seg_dir):
    """Build {class_id: [img_path, ...]} from segmentation directory.

    Expected layout: seg_dir/<class_name>/*.png  (or *.jpg)
    We derive class_id from sorted directory order to match names.txt.
    """
    class_dirs = sorted([
        d for d in glob.glob(os.path.join(seg_dir, "*"))
        if os.path.isdir(d)
    ])
    seg_map = {}
    for class_id, d in enumerate(class_dirs):
        paths = glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))
        if paths:
            seg_map[class_id] = paths
    return seg_map


def load_bg_paths(bg_dir):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(bg_dir, "**", ext), recursive=True)
    return paths


def random_color_jitter(img):
    """Random brightness, contrast, saturation, hue shift."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue ±10
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    # Saturation ×[0.7, 1.3]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    # Value (brightness) ×[0.6, 1.4]
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 1.4), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste_object(canvas, obj_bgr, obj_mask, x, y):
    """Paste obj onto canvas at top-left (x, y). Returns bbox (x1,y1,x2,y2) or None."""
    h, w = obj_bgr.shape[:2]
    ch, cw = canvas.shape[:2]

    # Clip to canvas bounds
    x1 = max(x, 0); y1 = max(y, 0)
    x2 = min(x + w, cw); y2 = min(y + h, ch)
    if x2 <= x1 or y2 <= y1:
        return None

    ox1 = x1 - x; oy1 = y1 - y
    ox2 = ox1 + (x2 - x1); oy2 = oy1 + (y2 - y1)

    mask_roi = obj_mask[oy1:oy2, ox1:ox2]
    obj_roi  = obj_bgr [oy1:oy2, ox1:ox2]

    mask_3ch = mask_roi[:, :, np.newaxis] / 255.0
    canvas[y1:y2, x1:x2] = (
        obj_roi * mask_3ch + canvas[y1:y2, x1:x2] * (1 - mask_3ch)
    ).astype(np.uint8)

    return (x1, y1, x2, y2)


def make_mask(img):
    """Extract foreground mask from segmentation image.

    Assumes background is white (>240 on all channels) or transparent (PNG alpha).
    """
    if img.shape[2] == 4:
        # PNG with alpha channel
        return img[:, :, 3]
    # White background removal
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def augment_one(seg_map, bg_paths, canvas_size=(640, 640), max_objects=4,
                min_scale=0.15, max_scale=0.45):
    bg_path = random.choice(bg_paths)
    bg = cv2.imread(bg_path)
    if bg is None:
        bg = np.ones((*canvas_size, 3), dtype=np.uint8) * 200
    bg = cv2.resize(bg, canvas_size)
    canvas = bg.copy()

    yolo_labels = []
    n_objects = random.randint(1, max_objects)

    # Randomly pick which classes to paste (allow repeats for stacked products)
    class_ids = random.choices(list(seg_map.keys()), k=n_objects)

    for class_id in class_ids:
        seg_path = random.choice(seg_map[class_id])
        seg_img  = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg_img is None:
            continue
        if seg_img.ndim == 2:
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)

        mask = make_mask(seg_img)
        if seg_img.shape[2] == 4:
            seg_bgr = seg_img[:, :, :3]
        else:
            seg_bgr = seg_img

        # Random scale
        scale = random.uniform(min_scale, max_scale)
        new_w = int(canvas_size[0] * scale)
        new_h = int(seg_bgr.shape[0] * new_w / seg_bgr.shape[1])
        seg_bgr = cv2.resize(seg_bgr, (new_w, new_h))
        mask    = cv2.resize(mask,    (new_w, new_h))

        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
        seg_bgr = cv2.warpAffine(seg_bgr, M, (new_w, new_h))
        mask    = cv2.warpAffine(mask,    M, (new_w, new_h))

        # Color jitter
        seg_bgr = random_color_jitter(seg_bgr)

        # Random position (allow partial occlusion at edges)
        x = random.randint(-new_w // 4, canvas_size[0] - new_w * 3 // 4)
        y = random.randint(-new_h // 4, canvas_size[1] - new_h * 3 // 4)

        bbox = paste_object(canvas, seg_bgr, mask, x, y)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / canvas_size[0]
        cy = (y1 + y2) / 2 / canvas_size[1]
        bw = (x2 - x1)      / canvas_size[0]
        bh = (y2 - y1)      / canvas_size[1]
        if bw > 0.01 and bh > 0.01:
            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return canvas, yolo_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_dir",    required=True)
    parser.add_argument("--bg_dir",     required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--max_objects",type=int, default=4)
    parser.add_argument("--canvas_w",   type=int, default=640)
    parser.add_argument("--canvas_h",   type=int, default=640)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)

    seg_map  = load_seg_images(args.seg_dir)
    bg_paths = load_bg_paths(args.bg_dir)
    print(f"Classes: {len(seg_map)}, Backgrounds: {len(bg_paths)}")

    canvas_size = (args.canvas_w, args.canvas_h)
    for i in range(args.num_images):
        canvas, labels = augment_one(
            seg_map, bg_paths,
            canvas_size=canvas_size,
            max_objects=args.max_objects,
        )
        name = f"aug_{i:05d}"
        cv2.imwrite(os.path.join(args.out_dir, "images", name + ".jpg"), canvas)
        with open(os.path.join(args.out_dir, "labels", name + ".txt"), "w") as f:
            f.write("\n".join(labels))

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{args.num_images} done")

    print("Augmentation complete.")


if __name__ == "__main__":
    main()
