"""
YOLO11 Knowledge Distillation Trainer.

Teacher(YOLOv7)가 생성한 GT box별 soft label(.npy)을 이용해
YOLO11 student 학습 시 classification loss에 KL divergence를 추가한다.

KD Loss:
  L_total = (1-alpha)*L_det + alpha * tau^2 * KL(teacher_soft || student_soft)

Usage:
    from kd_trainer import KDTrainer
    trainer = KDTrainer(
        soft_label_dir="~/Dataset/soft_labels",
        alpha=0.5, tau=4.0,
        model="yolo11m.pt", data="data/custom.yaml",
        epochs=100, batch=16, imgsz=640,
    )
    trainer.train()
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss


class KDDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss에 KD soft-label KL loss를 추가.

    __call__ 시그니처는 동일 (preds, batch).
    batch에 'soft_labels' 키가 있으면 KL divergence term 추가.
    """

    def __init__(self, model, alpha=0.5, tau=4.0, **kwargs):
        super().__init__(model, **kwargs)
        self.alpha = alpha
        self.tau   = tau

    def __call__(self, preds, batch):
        base_loss, loss_items = super().__call__(preds, batch)

        if "soft_labels" not in batch or batch["soft_labels"] is None:
            return base_loss, loss_items

        # soft_labels: list of tensors (one per image), each [N_gt_i, nc]
        soft_labels_batch = batch["soft_labels"]
        device = base_loss.device

        # --- student class prediction 수집 ---
        # ultralytics v8 loss 내부에서 이미 GT→anchor 할당이 완료된 뒤이므로
        # pred_scores (sigmoid 후 [B, num_anchors, nc])를 직접 가져온다.
        # forward의 두 번째 원소 (aux output 없을 때 preds[1])가 raw [B, nc+4+reg, A] 텐서.
        if isinstance(preds, (list, tuple)):
            raw = preds[1] if len(preds) > 1 else preds[0]
        else:
            raw = preds

        # raw: [B, nc+reg_max*4, A] — ultralytics detect head format
        # cls scores are the last nc channels
        nc = self.nc
        # [B, A, nc]
        cls_logits = raw[:, -nc:, :].permute(0, 2, 1)  # [B, A, nc]

        kd_loss = torch.tensor(0.0, device=device)
        total_gt = 0

        for b_idx in range(len(soft_labels_batch)):
            soft = soft_labels_batch[b_idx]  # [N_gt, nc] numpy or tensor
            if soft is None or (hasattr(soft, '__len__') and len(soft) == 0):
                continue
            if not isinstance(soft, torch.Tensor):
                soft = torch.tensor(soft, dtype=torch.float32)
            soft = soft.to(device)               # [N_gt, nc]
            n_gt = soft.shape[0]
            if n_gt == 0:
                continue

            # batch의 GT 인덱스로 해당 이미지 GT box들에 할당된 anchor 찾기
            # batch["batch_idx"] == b_idx인 행들
            gt_mask = batch["batch_idx"] == b_idx
            if gt_mask.sum() == 0:
                continue

            # 각 GT box에 가장 가까운 anchor를 cls_logits에서 가져오기.
            # 간단한 근사: GT box center를 feature map 좌표로 변환 후 nearest anchor 선택.
            # 정확한 TAL assignment는 손대지 않고, center-distance 기반 근사를 사용.
            gt_bboxes = batch["bboxes"][gt_mask]  # [n_gt_local, 4] xywhn
            img_h = img_w = self.imgsz if hasattr(self, "imgsz") else 640

            # GT center in [0,1]
            cx = gt_bboxes[:, 0]  # [n_gt_local]
            cy = gt_bboxes[:, 1]

            # Anchor center positions (ultralytics는 feature map grid center가 anchor)
            # cls_logits[b_idx]: [A, nc] — A = sum over scales of H*W*na
            # 근사: anchor center를 직접 구하는 대신, top-nc를 이용한 soft matching
            # 실용적 접근: batch["gt_assign"] 또는 b_idx image에 할당된 anchors를
            #   batch["targets"] 에서 추출한다.
            # ultralytics 내부 self.assigner가 forward_targets를 저장하면 좋지만
            # API가 버전마다 다르므로, student의 top-1 prediction per GT box를 사용.

            # student cls pred for this image: [A, nc]
            student_cls = cls_logits[b_idx]  # [A, nc]

            # GT별로: student의 class score 중 teacher soft label의 argmax class에
            # 가장 높은 score를 가진 anchor를 target anchor로 선택 (nearest top-1)
            teacher_cls = soft[:n_gt].float()                 # [n_gt, nc]
            teacher_hard = teacher_cls.argmax(dim=1)          # [n_gt]

            # 각 GT에 대해 teacher가 지목한 class에서 가장 score 높은 anchor index
            # [A, nc] → [n_gt, A]: teacher class에서의 student score
            student_for_teacher = student_cls[:, teacher_hard].T  # [n_gt, A]
            best_anchor = student_for_teacher.argmax(dim=1)        # [n_gt]

            student_probs = student_cls[best_anchor].sigmoid()     # [n_gt, nc]

            # KL divergence: sum_c p_t * log(p_t / p_s)
            # = F.kl_div(log(p_s), p_t, reduction='batchmean') * n
            kl = F.kl_div(
                torch.log(student_probs + 1e-8),
                teacher_cls,
                reduction="sum",
            )
            kd_loss = kd_loss + kl
            total_gt += n_gt

        if total_gt > 0:
            kd_loss = kd_loss / total_gt * (self.tau ** 2)

        combined = (1.0 - self.alpha) * base_loss + self.alpha * kd_loss
        return combined, loss_items


class KDDataset(torch.utils.data.Dataset):
    """
    ultralytics YOLODataset를 래핑해서 soft_label 필드를 추가한다.
    """

    def __init__(self, base_dataset, soft_label_dir):
        self.base = base_dataset
        self.soft_dir = Path(soft_label_dir).expanduser()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # img_path에서 stem 추출
        img_path = getattr(self.base, "im_files", None)
        stem = Path(img_path[idx]).stem if img_path else None
        soft = None
        if stem:
            npy_path = self.soft_dir / (stem + ".npy")
            if npy_path.exists():
                soft = np.load(str(npy_path))
        item["soft_label"] = soft
        return item


class KDTrainer(DetectionTrainer):
    """
    soft_label_dir와 KD 하이퍼파라미터를 받아 KD 학습을 수행.
    """

    def __init__(self, soft_label_dir, alpha=0.5, tau=4.0, **kwargs):
        self.soft_label_dir = Path(soft_label_dir).expanduser()
        self.kd_alpha = alpha
        self.kd_tau   = tau
        super().__init__(**kwargs)

    def get_dataloader(self, dataset_path, batch_size, rank, mode):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode == "train":
            loader.dataset.__class__ = type(
                "KDWrapped",
                (KDDataset, loader.dataset.__class__),
                {},
            )
            loader.dataset.soft_dir = self.soft_label_dir
            loader.dataset.base     = loader.dataset
        return loader

    def init_criterion(self):
        loss_fn = KDDetectionLoss(
            self.model,
            alpha=self.kd_alpha,
            tau=self.kd_tau,
        )
        return loss_fn

    @staticmethod
    def collate_fn(batch):
        """ultralytics default collate + soft_labels 처리."""
        from ultralytics.data.dataset import YOLODataset
        result = YOLODataset.collate_fn(batch)
        soft_labels = [item.get("soft_label") for item in batch]
        result["soft_labels"] = soft_labels
        return result


def train_kd(
    model_path: str = "yolo11m.pt",
    data_yaml: str   = "data/custom.yaml",
    soft_label_dir: str = "~/Dataset/soft_labels",
    epochs: int     = 100,
    batch: int      = 16,
    imgsz: int      = 640,
    device: str     = "0",
    project: str    = "runs/kd",
    name: str       = "yolo11m_kd",
    alpha: float    = 0.5,
    tau: float      = 4.0,
    resume: bool    = False,
):
    """KD 학습 진입점."""
    model = YOLO(model_path)
    model.train(
        trainer=KDTrainer,
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        resume=resume,
        # KDTrainer kwargs
        soft_label_dir=soft_label_dir,
        alpha=alpha,
        tau=tau,
        # standard hyperparams
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=10,
        augment=True,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="yolo11m.pt")
    p.add_argument("--data",           required=True)
    p.add_argument("--soft_label_dir", default="~/Dataset/soft_labels")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--device",             default="0")
    p.add_argument("--project",            default="runs/kd")
    p.add_argument("--name",               default="yolo11m_kd")
    p.add_argument("--alpha",    type=float, default=0.5)
    p.add_argument("--tau",      type=float, default=4.0)
    p.add_argument("--resume",   action="store_true")
    args = p.parse_args()
    train_kd(
        model_path=args.model,
        data_yaml=args.data,
        soft_label_dir=args.soft_label_dir,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        alpha=args.alpha,
        tau=args.tau,
        resume=args.resume,
    )
