"""
RF-DETR raw forward pass를 직접 호출해서 top-2 class margin(1등-2등 확신도 차이)
까지 뽑는 추론 모듈. model.predict()는 top-1만 반환하고 나머지 클래스 확률을
버리기 때문에(rfdetr/models/postprocess.py의 PostProcess._select_topk 확인됨),
postprocess() 호출 전 단계(pred_logits)에 직접 접근해서 계산.

내부 API(model.model, model.model.postprocess, model.means/stds 등)에 의존하므로
rfdetr 라이브러리 버전이 바뀌면 깨질 수 있음 -- tools/validate_margin_infer.py로
predict()와 결과가 일치하는지 항상 먼저 검증할 것.

근거: rfdetr/detr.py의 RFDETRBase.predict() 소스 직접 확인(2026-07-04):
    predictions = self.model.model(batch_tensor)          # (또는 inference_model, optimized 시)
    results = self.model.postprocess(predictions, target_sizes=target_sizes)
rfdetr/models/postprocess.py의 PostProcess._select_topk():
    prob = out_logits.sigmoid()   # 클래스별 독립 시그모이드 (소프트맥스 아님)
    topk_boxes = topk_indexes // out_logits.shape[2]   # 쿼리 인덱스
    labels     = topk_indexes % out_logits.shape[2]    # 클래스 인덱스
"""

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF


def _preprocess(pil_images, model_wrapper):
    """RFDETRBase.predict()와 정확히 동일한 순서: to_tensor -> device로 이동 -> resize -> normalize.
    (2026-07-04: 원래 CPU에서 resize하고 나중에 device로 옮겼었는데, predict()
    소스 확인 결과 device 이동이 resize보다 먼저였음 -- CPU/GPU bilinear resize
    커널의 부동소수점 오차가 다를 수 있어 순서를 정확히 맞춤. validate_margin_infer.py
    에서 confidence가 최대 0.08까지 벌어졌던 원인으로 의심됨.)
    """
    tensors = [TF.to_tensor(img).to(model_wrapper.model.device) for img in pil_images]
    resolution = model_wrapper.model.resolution
    resize_to = [resolution, resolution]
    batch = torch.stack([TF.resize(t, resize_to) for t in tensors])
    batch = TF.normalize(batch, model_wrapper.means, model_wrapper.stds)
    return batch


def _raw_forward(model_wrapper, batch_tensor):
    """predict()와 동일한 raw forward 경로 (optimized/non-optimized 분기 포함)."""
    if model_wrapper._is_optimized_for_inference:
        predictions = model_wrapper.model.inference_model(
            batch_tensor.to(dtype=model_wrapper._optimized_dtype)
        )
    else:
        predictions = model_wrapper.model.model(batch_tensor)
    if isinstance(predictions, tuple):
        predictions = {"pred_logits": predictions[1], "pred_boxes": predictions[0]}
    return predictions


def infer_rfdetr_with_margin(model_wrapper, frames, conf_thres=0.4, device="cuda:0",
                              num_select=300):
    """
    frames: List[np.ndarray | None] -- 카메라별 BGR 프레임
    반환: per_cam 리스트. 각 detection dict: class_id, confidence, bbox, margin
          margin = 그 query의 top1 sigmoid확률 - top2 sigmoid확률 (0에 가까울수록 헷갈림)
    """
    valid_idx = [i for i, f in enumerate(frames) if f is not None]
    per_cam = [None] * len(frames)
    if not valid_idx:
        return per_cam

    pil_images = [
        PIL.Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        for i in valid_idx
    ]
    orig_sizes = [(img.height, img.width) for img in pil_images]

    with torch.no_grad():
        batch_tensor = _preprocess(pil_images, model_wrapper)
        predictions = _raw_forward(model_wrapper, batch_tensor)

        out_logits = predictions["pred_logits"]  # (B, Q, C)
        prob = out_logits.sigmoid()
        B, Q, C = prob.shape

        # postprocess()가 내부적으로 하는 것과 동일한 top-k 선택을 직접 재현
        # (postprocess()는 pixel-scale 박스 계산용으로, 우리 topk_boxes(쿼리 인덱스)는
        #  postprocess() 바깥으로 안 나오기 때문에 별도 계산 필요 -- 둘 다 같은
        #  out_logits에 대한 결정론적 계산이라 순서가 동일하게 나옴)
        logits_flat = prob.view(B, -1)
        k = min(num_select, logits_flat.shape[1])
        topk_values, topk_indexes = torch.topk(logits_flat, k, dim=1)
        topk_query = topk_indexes // C
        topk_label = topk_indexes % C

        target_sizes = torch.tensor(orig_sizes, device=out_logits.device)
        results = model_wrapper.model.postprocess(predictions, target_sizes=target_sizes)

        for b, orig_idx in enumerate(valid_idx):
            dets = []
            scores_b = results[b]["scores"]
            labels_b = results[b]["labels"]
            boxes_b = results[b]["boxes"]
            n = scores_b.shape[0]
            for i in range(n):
                score = float(scores_b[i])
                if score < conf_thres:
                    continue
                query_idx = int(topk_query[b, i])
                full_probs = prob[b, query_idx, :]  # (C,)
                top2_vals, _ = torch.topk(full_probs, 2)
                margin = float(top2_vals[0] - top2_vals[1])
                x1, y1, x2, y2 = [float(v) for v in boxes_b[i]]
                dets.append({
                    "class_id": int(labels_b[i]),
                    "confidence": score,
                    "bbox": [x1, y1, x2, y2],
                    "margin": margin,
                })
            per_cam[orig_idx] = dets

    return per_cam
