"""
RF-DETR 추론 모듈 — infer_batch와 동일한 인터페이스 제공
"""

import cv2
import numpy as np
import PIL.Image


def load_rfdetr(weights: str, num_classes: int, device: str):
    from rfdetr import RFDETRBase
    # device는 내부적으로 자동 처리됨 (CUDA 사용 가능 시 자동)
    model = RFDETRBase(
        pretrain_weights=weights if weights else None,
        num_classes=num_classes,
        resolution=672,  # 56(=patch14*win4) 배수 필요; 640은 불가
    )
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)

        result = model.predict(pil_img, threshold=conf_thres)

        dets = []
        if result is not None and len(result.class_id) > 0:
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
