#!/bin/bash
# RF-DETR + SAM2 환경 구성 (서버용)
# 실행: bash setup_rfdetr_env.sh

set -e

ENV_NAME="rfdetr"
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -1)
echo "Detected CUDA: ${CUDA_VER}"

# conda env 생성
conda create -n $ENV_NAME python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# CUDA 11.x → cu118 wheel, CUDA 12.x → cu121 wheel
if [[ "$CUDA_VER" == 12* ]]; then
    TORCH_IDX="https://download.pytorch.org/whl/cu121"
else
    TORCH_IDX="https://download.pytorch.org/whl/cu118"
fi
pip install torch torchvision --index-url $TORCH_IDX

# RF-DETR (Roboflow)
pip install rfdetr

# SAM2 (Meta)
pip install git+https://github.com/facebookresearch/sam2.git

# SAM2 체크포인트 다운로드
mkdir -p ~/checkpoints/sam2
cd ~/checkpoints/sam2
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

pip install opencv-python-headless numpy pillow tqdm pycocotools

echo "Done. conda activate $ENV_NAME"
