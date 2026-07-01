"""
SAM2 + YOLOv7로 테스트 영상에서 도메인 특화 학습 데이터 추출

흐름:
  1. cam0~4 영상에서 N프레임 간격으로 keyframe 추출
  2. YOLOv7로 bbox 감지 → SAM2에 bbox prompt로 전달
  3. SAM2가 돌려준 정밀 mask의 tight bbox를 crop → COCO 어노테이션 저장

Usage (rfdetr conda 환경에서):
    python tools/sam2_video_label.py \
        --videos   ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
                   ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
                   ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
                   ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
                   ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4 \
        --weights  ~/Dataset/yolov7_custom.pt \
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
from PIL import Image

# YOLOv7 추론 (기존 파이프라인 재사용)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos",    nargs="+", required=True)
    p.add_argument("--weights",   default=os.path.expanduser("~/Dataset/yolov7_custom.pt"))
    p.add_argument("--names",     default="data/names.txt")
    p.add_argument("--sam2_ckpt", default=os.path.expanduser("~/checkpoints/sam2/sam2.1_hiera_large.pt"))
    p.add_argument("--out_dir",   default="data/coco_rfdetr")
    p.add_argument("--interval",  type=int, default=30, help="프레임 샘플링 간격")
    p.add_argument("--conf",      type=float, default=0.3)
    p.add_argument("--device",    default="0")
    return p.parse_args()


def load_yolov7(weights, device):
    yolov7_root = str(Path.home() / "yolov7")
    if yolov7_root not in sys.path:
        sys.path.insert(0, yolov7_root)
    from utils.general import non_max_suppression
    import torch.nn as nn
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().fuse().eval().to(device)
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    return model, non_max_suppression


def preprocess(frame, size=640):
    img = cv2.resize(frame, (size, size))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0


def detect_yolov7(model, nms_fn, frame, conf, device):
    t = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = nms_fn(model(t)[0], conf, 0.45)[0]
    H, W = frame.shape[:2]
    dets = []
    if pred is not None and len(pred):
        for *xyxy, c, cls in pred.cpu().numpy():
            x1, y1, x2, y2 = xyxy
            # 640 좌표 → 원본 좌표
            x1 = x1 / 640 * W; x2 = x2 / 640 * W
            y1 = y1 / 640 * H; y2 = y2 / 640 * H
            dets.append((int(cls), float(c), [x1, y1, x2, y2]))
    return dets


def load_sam2(ckpt_path, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # 모델 설정은 large 고정
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(cfg, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(sam2)
    return predictor


def refine_bbox_with_sam2(predictor, frame_rgb, bboxes_xyxy):
    """SAM2로 bbox prompt → 정밀 mask → tight bbox 반환"""
    predictor.set_image(frame_rgb)
    refined = []
    for bbox in bboxes_xyxy:
        x1, y1, x2, y2 = bbox
        input_box = np.array([[x1, y1, x2, y2]])
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        mask = masks[0].astype(bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            refined.append(bbox)
            continue
        refined.append([float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())])
    return refined


def main():
    args = parse_args()
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    with open(args.names) as f:
        names = [l.strip() for l in f if l.strip()]

    out_dir   = Path(args.out_dir)
    img_out   = out_dir / "images" / "video_domain"
    ann_dir   = out_dir / "annotations"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLOv7...")
    yolo, nms_fn = load_yolov7(args.weights, device)

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

            dets = detect_yolov7(yolo, nms_fn, frame, args.conf, device)
            if not dets:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes    = [d[2] for d in dets]
            refined   = refine_bbox_with_sam2(sam2_pred, frame_rgb, bboxes)

            fname = f"{cam_name}_f{frame_idx:06d}.jpg"
            dst   = img_out / fname
            cv2.imwrite(str(dst), frame)

            H, W = frame.shape[:2]
            img_id = len(images) + 1
            images.append({
                "id": img_id,
                "file_name": f"video_domain/{fname}",
                "width": W, "height": H,
            })

            for (cls_id, conf, _), (rx1, ry1, rx2, ry2) in zip(dets, refined):
                bw, bh = rx2 - rx1, ry2 - ry1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(rx1, 2), round(ry1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                    "score": round(conf, 3),
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
    print("이제 yolo_to_coco.py 출력과 이 데이터를 합쳐서 RF-DETR 학습에 사용하세요.")


if __name__ == "__main__":
    main()
