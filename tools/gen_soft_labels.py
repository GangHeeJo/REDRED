"""
YOLOv7 teacher로 훈련 이미지의 GT box별 soft label 생성.

각 훈련 이미지에 대해:
  - GT 라벨 파일 읽기 (YOLO format: cls cx cy w h)
  - YOLOv7 raw 추론 (pre-NMS)으로 각 GT box와 최고 IoU anchor의 class probs 추출
  - {img_stem}.npy 로 저장: shape [N_gt, 60] — N_gt개 GT box × 60 class 확률

Usage:
    python tools/gen_soft_labels.py \
        --train_txt ~/yolov7/data/train.txt \
        --weights   ~/Dataset/yolov7_custom.pt \
        --out_dir   ~/Dataset/soft_labels \
        --device    0
"""
import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path.home() / "yolov7"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_txt", required=True)
    p.add_argument("--weights",   required=True)
    p.add_argument("--out_dir",   required=True)
    p.add_argument("--device",    default="0")
    p.add_argument("--img_size",  type=int, default=640)
    p.add_argument("--conf",      type=float, default=0.01,
                   help="낮게 설정해서 모든 anchor 후보 확보")
    return p.parse_args()


def box_iou_xyxy(a, b):
    """a: [N,4], b: [M,4] → IoU [N,M]"""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = torch.max(ax1[:, None], bx1[None, :])
    inter_y1 = torch.max(ay1[:, None], by1[None, :])
    inter_x2 = torch.min(ax2[:, None], bx2[None, :])
    inter_y2 = torch.min(ay2[:, None], by2[None, :])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def xywhn_to_xyxy(boxes, w, h):
    """YOLO norm xywh → pixel xyxy"""
    cx, cy, bw, bh = boxes[:, 0]*w, boxes[:, 1]*h, boxes[:, 2]*w, boxes[:, 3]*h
    return torch.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], dim=1)


def load_label(label_path, img_w, img_h):
    """GT 라벨 파일 → (cls_ids [N], boxes_xyxy [N,4])"""
    if not os.path.exists(label_path):
        return None, None
    data = np.loadtxt(label_path).reshape(-1, 5)
    if len(data) == 0:
        return None, None
    cls_ids = torch.tensor(data[:, 0], dtype=torch.long)
    boxes   = xywhn_to_xyxy(torch.tensor(data[:, 1:], dtype=torch.float32), img_w, img_h)
    return cls_ids, boxes


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # PyTorch 버전 호환성 패치: recompute_scale_factor 속성 제거된 이후 버전 대응
    import torch.nn.functional as F_nn
    torch.nn.Upsample.forward = lambda self, x: F_nn.interpolate(
        x, self.size, self.scale_factor, self.mode, self.align_corners
    )

    print("Loading YOLOv7 teacher...")
    ckpt = torch.load(args.weights, map_location=device)
    model = ckpt["model"].float().eval().to(device)
    num_classes = model.yaml.get("nc", 60)
    print(f"  num_classes={num_classes}")

    with open(args.train_txt) as f:
        img_paths = [l.strip() for l in f if l.strip()]

    from tqdm import tqdm
    import cv2

    skipped = 0
    for img_path in tqdm(img_paths, desc="Generating soft labels"):
        stem = Path(img_path).stem
        out_path = os.path.join(args.out_dir, stem + ".npy")
        if os.path.exists(out_path):
            continue

        # 라벨 경로 (images → labels, .jpg/.png → .txt)
        label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        ih, iw = img.shape[:2]

        cls_ids, gt_boxes = load_label(label_path, iw, ih)
        if cls_ids is None:
            np.save(out_path, np.zeros((0, num_classes), dtype=np.float32))
            continue

        # 이미지 전처리
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size))
        inp = torch.from_numpy(img_resized).permute(2,0,1).float().div(255).unsqueeze(0).to(device)

        with torch.no_grad():
            raw = model(inp)

        # YOLOv7 eval 모드 출력: (z_cat, x_list)
        #   z_cat: [1, N_all, 5+nc]  — 모든 scale 합쳐진 상태, sigmoid 이미 적용됨
        #     [:, :2]  = cx, cy (pixels, img_size 기준)
        #     [:, 2:4] = w, h   (pixels, img_size 기준)
        #     [:, 4]   = obj_conf (sigmoid)
        #     [:, 5:]  = class_probs (sigmoid)
        z_cat = raw[0] if isinstance(raw, (list, tuple)) else raw
        pred_all = z_cat[0]  # [N_all, 5+nc]

        scale_x, scale_y = args.img_size / iw, args.img_size / ih
        obj_conf = pred_all[:, 4]  # 이미 sigmoid 됨
        mask = obj_conf > args.conf
        if mask.sum() == 0:
            # teacher가 아무것도 못 잡으면 uniform soft label
            soft = np.full((len(cls_ids), num_classes), 1.0/num_classes, dtype=np.float32)
            np.save(out_path, soft)
            continue

        pred_filt = pred_all[mask]  # [M, 5+nc]
        cx, cy = pred_filt[:, 0], pred_filt[:, 1]
        w,  h  = pred_filt[:, 2], pred_filt[:, 3]
        pred_boxes = torch.stack([
            (cx - w/2) / scale_x,
            (cy - h/2) / scale_y,
            (cx + w/2) / scale_x,
            (cy + h/2) / scale_y,
        ], dim=1)
        pred_probs = pred_filt[:, 5:]  # [M, nc] 이미 sigmoid 됨
        gt_boxes   = gt_boxes.to(device)

        # GT box마다 best IoU anchor의 class probs 추출
        iou = box_iou_xyxy(gt_boxes, pred_boxes)  # [N_gt, N_pred]
        best_idx = iou.argmax(dim=1)               # [N_gt]
        soft_labels = pred_probs[best_idx].cpu().numpy()  # [N_gt, nc]

        # temperature scaling (τ=4): sharpen/soften
        tau = 4.0
        logits = np.log(soft_labels + 1e-8) / tau
        soft_labels = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        np.save(out_path, soft_labels.astype(np.float32))

    print(f"Done. skipped={skipped}")


if __name__ == "__main__":
    main()
