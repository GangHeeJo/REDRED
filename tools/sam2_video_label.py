"""
SAM2 + RF-DETR로 테스트 영상에서 도메인 특화 학습 데이터 추출

흐름:
  1. cam0~4 영상에서 N프레임 간격으로 keyframe 추출
  2. RF-DETR로 bbox 감지 → SAM2에 bbox prompt로 전달
  3. SAM2가 돌려준 정밀 mask의 tight bbox를 COCO 어노테이션으로 저장

Usage (rfdetr conda 환경에서):
    python tools/sam2_video_label.py \
        --videos   ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 ... \
        --weights  runs/rfdetr/checkpoint_best_total.pth \
        --names    data/names.txt \
        --sam2_ckpt ~/checkpoints/sam2/sam2.1_hiera_large.pt \
        --out_dir  data/coco_rfdetr \
        --interval 30
"""

import argparse
import json
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from infer_rfdetr import load_rfdetr, infer_rfdetr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos",    nargs="+", required=True)
    p.add_argument("--weights",   default="runs/rfdetr/checkpoint_best_total.pth")
    p.add_argument("--names",     default="data/names.txt")
    p.add_argument("--sam2_ckpt", default=os.path.expanduser("~/checkpoints/sam2/sam2.1_hiera_large.pt"))
    p.add_argument("--out_dir",   default="data/coco_rfdetr")
    p.add_argument("--interval",  type=int, default=30)
    p.add_argument("--conf",      type=float, default=0.3)
    p.add_argument("--device",    default="0")
    return p.parse_args()


def load_sam2(ckpt_path, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(cfg, ckpt_path, device=device)
    return SAM2ImagePredictor(sam2)


def refine_bbox_with_sam2(predictor, frame_rgb, bboxes_xyxy):
    predictor.set_image(frame_rgb)
    refined = []
    for bbox in bboxes_xyxy:
        x1, y1, x2, y2 = bbox
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([[x1, y1, x2, y2]]),
            multimask_output=False,
        )
        mask = masks[0].astype(bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            refined.append(bbox)
        else:
            refined.append([float(xs.min()), float(ys.min()),
                            float(xs.max()), float(ys.max())])
    return refined


def main():
    args = parse_args()
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    with open(args.names) as f:
        names = [l.strip() for l in f if l.strip()]

    out_dir = Path(args.out_dir)
    img_out = out_dir / "images" / "video_domain"
    ann_dir = out_dir / "annotations"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RF-DETR...")
    rfdetr = load_rfdetr(args.weights, num_classes=len(names), device=device)

    print("Loading SAM2...")
    sam2_pred = load_sam2(args.sam2_ckpt, device)

    categories = [{"id": i, "name": n, "supercategory": "product"}
                  for i, n in enumerate(names)]
    images, annotations = [], []
    ann_id = 1

    for vid_path in args.videos:
        cap = cv2.VideoCapture(vid_path)
        cam_name = Path(vid_path).parent.name
        frame_idx = 0

        pbar = tqdm(desc=cam_name, total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            if frame_idx % args.interval != 0:
                frame_idx += 1
                continue

            # RF-DETR 추론 (단일 프레임을 리스트로 전달)
            per_cam = infer_rfdetr(rfdetr, [frame], conf_thres=args.conf, device=device)
            dets = per_cam[0] or []
            if not dets:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes    = [d["bbox"] for d in dets]
            refined   = refine_bbox_with_sam2(sam2_pred, frame_rgb, bboxes)

            fname  = f"{cam_name}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(img_out / fname), frame)

            H, W   = frame.shape[:2]
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": f"video_domain/{fname}",
                           "width": W, "height": H})

            for det, (rx1, ry1, rx2, ry2) in zip(dets, refined):
                bw, bh = rx2 - rx1, ry2 - ry1
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": det["class_id"],
                    "bbox": [round(rx1, 2), round(ry1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                    "score": round(det["confidence"], 3),
                })
                ann_id += 1

            frame_idx += 1
        pbar.close()
        cap.release()

    out_json = ann_dir / "instances_video_domain.json"
    with open(out_json, "w") as f:
        json.dump({"images": images, "annotations": annotations,
                   "categories": categories}, f)
    print(f"\n{len(images)} frames, {len(annotations)} annotations → {out_json}")


if __name__ == "__main__":
    main()
