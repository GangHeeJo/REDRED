"""
Cut & Paste Augmentation for unmanned vending machine dataset.

추가된 증강 기법 목록:
    1. Cut & Paste       : 배경 이미지에 세그멘테이션 상품 이미지를 붙임 (기존)
       - Random scale, rotation ±15°, color jitter, random position 포함
    2. Horizontal Flip   : 좌우 반전 + bbox x 좌표 보정 (확률 0.5)
    3. Perspective Warp  : 원근 변환 → 5개 카메라 각도 차이 모사 (확률 0.5)
    4. Random Erasing    : 화면 일부를 랜덤 노이즈로 가림 → occlusion 대응
    5. Blur + Noise      : Gaussian blur + Gaussian noise → 실제 카메라 품질 모사

Usage:
    python cut_paste_aug.py \
        --seg_dir ~/Dataset/2.backsub_images_100 \
        --bg_dir  ~/Dataset/3.Background_Images \
        --out_dir ~/Dataset/augmented \
        --num_images 5000 \
        --max_objects 4

    # 특정 증강 비활성화:
    python cut_paste_aug.py ... --no_flip --no_perspective

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


# ---------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------

def load_seg_images(seg_dir):
    """Build {class_id: [img_path, ...]} from segmentation directory.

    Expected layout: seg_dir/<class_name>/*.png  (or *.jpg)
    class_id는 디렉토리 정렬 순서 기준 (names.txt와 동일하게 맞춰야 함).
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


# ---------------------------------------------------------------
# 기존 증강: Cut & Paste 구성 요소
# ---------------------------------------------------------------

def random_color_jitter(img):
    """Random brightness, contrast, saturation, hue shift."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 1.4), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def make_mask(img):
    """Extract foreground mask from segmentation image.

    흰 배경(>240) 또는 PNG alpha 채널 기준으로 전경 마스크 추출.
    """
    if img.shape[2] == 4:
        return img[:, :, 3]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def paste_object(canvas, obj_bgr, obj_mask, x, y):
    """Paste obj onto canvas at top-left (x, y). Returns bbox (x1,y1,x2,y2) or None."""
    h, w = obj_bgr.shape[:2]
    ch, cw = canvas.shape[:2]

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


# ---------------------------------------------------------------
# 추가 증강 기법 2: Horizontal Flip
# ---------------------------------------------------------------

def random_hflip(canvas, labels, prob=0.5):
    """
    [추가] 좌우 반전.
    bbox의 cx 좌표를 1 - cx 로 보정.
    무인판매대 카메라가 정면/측면 모두 촬영하므로
    반전만으로도 데이터 다양성이 크게 늘어남.
    """
    if random.random() > prob:
        return canvas, labels

    canvas = cv2.flip(canvas, 1)

    new_labels = []
    for label in labels:
        parts = label.split()
        cls = parts[0]
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        cx = 1.0 - cx
        new_labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return canvas, new_labels


# ---------------------------------------------------------------
# 추가 증강 기법 3: Perspective Warp
# ---------------------------------------------------------------

def random_perspective_warp(canvas, labels, max_shift_ratio=0.05, prob=0.5):
    """
    [추가] 원근 변환.
    캔버스 모서리 4개를 랜덤하게 조금씩 이동시켜 원근감 부여.
    5개 카메라가 서로 다른 각도에서 촬영하는 상황을 모사.

    max_shift_ratio: 캔버스 크기 대비 최대 이동 비율 (기본 5%)
    """
    if random.random() > prob:
        return canvas, labels

    h, w = canvas.shape[:2]
    d = int(min(h, w) * max_shift_ratio)

    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = pts1 + np.random.uniform(-d, d, pts1.shape).astype(np.float32)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    canvas = cv2.warpPerspective(canvas, M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT_101)

    new_labels = []
    for label in labels:
        parts = label.split()
        cls = parts[0]
        cx  = float(parts[1]) * w
        cy  = float(parts[2]) * h
        bw  = float(parts[3]) * w
        bh  = float(parts[4]) * h

        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2

        corners = np.array(
            [[x1, y1], [x2, y1], [x1, y2], [x2, y2]],
            dtype=np.float32
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

        x1n = np.clip(warped[:, 0].min(), 0, w)
        x2n = np.clip(warped[:, 0].max(), 0, w)
        y1n = np.clip(warped[:, 1].min(), 0, h)
        y2n = np.clip(warped[:, 1].max(), 0, h)

        cxn = (x1n + x2n) / 2 / w
        cyn = (y1n + y2n) / 2 / h
        bwn = (x2n - x1n) / w
        bhn = (y2n - y1n) / h

        if bwn > 0.01 and bhn > 0.01:
            new_labels.append(f"{cls} {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f}")

    return canvas, new_labels


# ---------------------------------------------------------------
# 추가 증강 기법 4: Random Erasing
# ---------------------------------------------------------------

def random_erasing(canvas, n_patches=3, min_ratio=0.02, max_ratio=0.12, prob=0.5):
    """
    [추가] 랜덤 영역을 노이즈로 가림.
    무인판매대에서 상품이 손/다른 상품에 부분적으로 가려지는 경우를 모사.
    bbox는 변경하지 않음 (가려진 상태에서도 인식하도록 학습).

    n_patches    : 최대 가리는 패치 수
    min/max_ratio: 캔버스 대비 패치 크기 비율
    """
    if random.random() > prob:
        return canvas

    h, w = canvas.shape[:2]
    for _ in range(random.randint(1, n_patches)):
        pw = int(w * random.uniform(min_ratio, max_ratio))
        ph = int(h * random.uniform(min_ratio, max_ratio))
        px = random.randint(0, max(0, w - pw))
        py = random.randint(0, max(0, h - ph))
        canvas[py:py + ph, px:px + pw] = np.random.randint(
            0, 256, (ph, pw, 3), dtype=np.uint8
        )

    return canvas


# ---------------------------------------------------------------
# 추가 증강 기법 5: Blur + Noise
# ---------------------------------------------------------------

def add_blur_and_noise(canvas, blur_prob=0.3, noise_prob=0.3):
    """
    [추가] Gaussian blur + Gaussian noise.
    실제 카메라의 초점 흔들림과 센서 노이즈를 모사.

    blur_prob : Gaussian blur 적용 확률
    noise_prob: Gaussian noise 적용 확률
    """
    if random.random() < blur_prob:
        ksize = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (ksize, ksize), 0)

    if random.random() < noise_prob:
        sigma = random.uniform(5, 20)
        noise = np.random.normal(0, sigma, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return canvas


# ---------------------------------------------------------------
# 메인 증강 함수
# ---------------------------------------------------------------

def augment_one(
    seg_map, bg_paths,
    canvas_size=(640, 640),
    max_objects=4,
    min_scale=0.15,
    max_scale=0.45,
    use_hflip=True,
    use_perspective=True,
    use_erasing=True,
    use_blur_noise=True,
):
    bg_path = random.choice(bg_paths)
    bg = cv2.imread(bg_path)
    if bg is None:
        bg = np.ones((*canvas_size[::-1], 3), dtype=np.uint8) * 200
    bg = cv2.resize(bg, canvas_size)
    canvas = bg.copy()

    yolo_labels = []
    n_objects = random.randint(1, max_objects)
    class_ids = random.choices(list(seg_map.keys()), k=n_objects)

    # --- Cut & Paste (기존) ---
    for class_id in class_ids:
        seg_path = random.choice(seg_map[class_id])
        seg_img  = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg_img is None:
            continue
        if seg_img.ndim == 2:
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)

        mask = make_mask(seg_img)
        seg_bgr = seg_img[:, :, :3] if seg_img.shape[2] == 4 else seg_img

        scale = random.uniform(min_scale, max_scale)
        new_w = int(canvas_size[0] * scale)
        new_h = int(seg_bgr.shape[0] * new_w / seg_bgr.shape[1])
        seg_bgr = cv2.resize(seg_bgr, (new_w, new_h))
        mask    = cv2.resize(mask,    (new_w, new_h))

        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
        seg_bgr = cv2.warpAffine(seg_bgr, M, (new_w, new_h))
        mask    = cv2.warpAffine(mask,    M, (new_w, new_h))

        seg_bgr = random_color_jitter(seg_bgr)

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

    # --- 추가 증강 적용 순서 ---
    # Flip → Perspective → Erasing → Blur/Noise
    if use_hflip:
        canvas, yolo_labels = random_hflip(canvas, yolo_labels)

    if use_perspective:
        canvas, yolo_labels = random_perspective_warp(canvas, yolo_labels)

    if use_erasing:
        canvas = random_erasing(canvas)

    if use_blur_noise:
        canvas = add_blur_and_noise(canvas)

    return canvas, yolo_labels


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cut & Paste augmentation with Flip, Perspective, Erasing, Blur+Noise"
    )
    parser.add_argument("--seg_dir",     required=True,
                        help="세그멘테이션 이미지 루트 (클래스별 하위 폴더)")
    parser.add_argument("--bg_dir",      required=True,
                        help="배경 이미지 폴더")
    parser.add_argument("--out_dir",     required=True,
                        help="출력 폴더 (images/, labels/ 자동 생성)")
    parser.add_argument("--num_images",  type=int, default=5000)
    parser.add_argument("--max_objects", type=int, default=4,
                        help="이미지당 최대 상품 수")
    parser.add_argument("--canvas_w",    type=int, default=640)
    parser.add_argument("--canvas_h",    type=int, default=640)

    # 증강 기법 개별 비활성화 플래그
    parser.add_argument("--no_flip",        action="store_true",
                        help="Horizontal Flip 비활성화")
    parser.add_argument("--no_perspective", action="store_true",
                        help="Perspective Warp 비활성화")
    parser.add_argument("--no_erasing",     action="store_true",
                        help="Random Erasing 비활성화")
    parser.add_argument("--no_blur",        action="store_true",
                        help="Blur + Noise 비활성화")

    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)

    seg_map  = load_seg_images(args.seg_dir)
    bg_paths = load_bg_paths(args.bg_dir)

    print(f"Classes    : {len(seg_map)}")
    print(f"Backgrounds: {len(bg_paths)}")
    print(f"Augmentations: "
          f"flip={'off' if args.no_flip else 'on'}, "
          f"perspective={'off' if args.no_perspective else 'on'}, "
          f"erasing={'off' if args.no_erasing else 'on'}, "
          f"blur+noise={'off' if args.no_blur else 'on'}")

    canvas_size = (args.canvas_w, args.canvas_h)

    for i in range(args.num_images):
        canvas, labels = augment_one(
            seg_map, bg_paths,
            canvas_size=canvas_size,
            max_objects=args.max_objects,
            use_hflip=not args.no_flip,
            use_perspective=not args.no_perspective,
            use_erasing=not args.no_erasing,
            use_blur_noise=not args.no_blur,
        )
        name = f"aug_{i:05d}"
        cv2.imwrite(os.path.join(args.out_dir, "images", name + ".jpg"), canvas)
        with open(os.path.join(args.out_dir, "labels", name + ".txt"), "w") as f:
            f.write("\n".join(labels))

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.num_images} done")

    print("Augmentation complete.")


if __name__ == "__main__":
    main()
