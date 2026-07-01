"""
RF-DETR 추론 모듈 — infer_batch와 동일한 인터페이스 제공
"""

import cv2
import numpy as np
import PIL.Image


def load_rfdetr(weights: str, num_classes: int, device: str):
    import torch
    from rfdetr import RFDETRBase
    model = RFDETRBase(
        pretrain_weights=weights if weights else None,
        num_classes=num_classes,
        resolution=672,  # 56(=patch14×win4) 배수; 640은 불가
    )
    # FP16 최적화 — T4 기준 ~8x 속도 향상 (rfdetr 경고 권장)
    try:
        model.optimize_for_inference(dtype=torch.float16)
        print("RF-DETR: FP16 inference enabled")
    except Exception as e:
        print(f"RF-DETR: FP16 최적화 실패 ({e}), FP32로 계속")
    return model


def _parse_detections(result) -> list:
    dets = []
    if result is None:
        return dets
    ids   = result.class_id
    confs = result.confidence
    boxes = result.xyxy
    if ids is None or len(ids) == 0:
        return dets
    for cls_id, conf, box in zip(ids, confs, boxes):
        x1, y1, x2, y2 = box
        dets.append({
            "class_id":   int(cls_id),
            "confidence": float(conf),
            "bbox":       [float(x1), float(y1), float(x2), float(y2)],
        })
    return dets


def infer_rfdetr(model, frames, conf_thres=0.4, device="cuda:0"):
    """
    frames: List[np.ndarray | None] — 5개 카메라 BGR 프레임
    반환:   per_cam 리스트 (infer_batch와 동일 형식)

    배치 predict 시도 → 실패 시 순차 fallback
    """
    valid_idx   = [i for i, f in enumerate(frames) if f is not None]
    pil_images  = [
        PIL.Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        for i in valid_idx
    ]

    per_cam = [None] * len(frames)
    if not pil_images:
        return per_cam

    # 배치 추론 시도
    try:
        results = model.predict(pil_images, threshold=conf_thres)
        # 결과가 리스트면 배치 성공
        if isinstance(results, list):
            for idx, result in zip(valid_idx, results):
                per_cam[idx] = _parse_detections(result)
            return per_cam
        # 단일 Detections 반환 → 배치 미지원, fallback
        per_cam[valid_idx[0]] = _parse_detections(results)
        for i in valid_idx[1:]:
            r = model.predict(pil_images[valid_idx.index(i)], threshold=conf_thres)
            per_cam[i] = _parse_detections(r)
        return per_cam
    except Exception:
        pass

    # 순차 fallback
    for i, pil_img in zip(valid_idx, pil_images):
        result = model.predict(pil_img, threshold=conf_thres)
        per_cam[i] = _parse_detections(result)
    return per_cam
