"""
RF-DETR 추론 모듈 — infer_batch와 동일한 인터페이스 제공

반환: per_cam = List[List[{"class_id", "confidence", "bbox"}]]
"""

import torch
import numpy as np
import cv2


def load_rfdetr(weights: str, num_classes: int, device: str):
    """
    RF-DETR 모델 로드.
    weights: fine-tuned .pth 경로 or None (COCO pretrained)
    """
    from rfdetr import RFDETRBase
    model = RFDETRBase(pretrain_weights=weights if weights else None,
                       num_classes=num_classes,
                       resolution=640)
    model.model.to(device)
    model.model.eval()
    return model


def infer_rfdetr(model, frames, conf_thres=0.4, device="cuda:0"):
    """
    frames: List[np.ndarray | None] — 5개 카메라 BGR 프레임
    반환:   per_cam 리스트 (infer_batch와 동일 형식)
    """
    per_cam = [None] * len(frames)
    for i, frame in enumerate(frames):
        if frame is None:
            continue
        # RF-DETR는 PIL RGB or numpy RGB 입력
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = __import__("PIL.Image", fromlist=["Image"]).Image.fromarray(rgb)

        with torch.no_grad():
            result = model.predict(pil_img, threshold=conf_thres)

        dets = []
        if result and hasattr(result, "class_id"):
            for cls_id, conf, box in zip(result.class_id,
                                          result.confidence,
                                          result.xyxy):
                x1, y1, x2, y2 = box
                dets.append({
                    "class_id":   int(cls_id),
                    "confidence": float(conf),
                    "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                })
        per_cam[i] = dets
    return per_cam
