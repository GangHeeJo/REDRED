"""
기존 학습된 YOLO11 weights에 SimAM을 forward hook으로 적용.
파라미터 0개이므로 재학습 불필요 — 어떤 YOLO11 가중치에도 바로 얹을 수 있음.

Usage (추론):
    from ultralytics import YOLO
    from src.patch_ultralytics import apply_simam_hooks

    model = YOLO("best.pt")
    apply_simam_hooks(model)
    # 이후 model.predict(...)는 P3/P4/P5에 SimAM이 적용된 채로 동작

Usage (fine-tuning, 선택적):
    model = YOLO("best.pt")
    apply_simam_hooks(model)
    model.train(data="data/custom.yaml", epochs=20, ...)
"""

from src.simam import SimAM

# 원본 YOLO11 레이어 인덱스 — P3/P4/P5 head 각 C3k2 출력 직후
_SIMAM_LAYER_INDICES = [16, 19, 22]


def apply_simam_hooks(model) -> None:
    """
    YOLO11 model.model.model[16/19/22] 출력에 SimAM forward hook 등록.
    파라미터 0개이므로 재학습 없이 기존 가중치 그대로 사용 가능.
    """
    _simam = SimAM()
    for idx in _SIMAM_LAYER_INDICES:
        model.model.model[idx].register_forward_hook(
            lambda m, inp, out, s=_simam: s(out)
        )
