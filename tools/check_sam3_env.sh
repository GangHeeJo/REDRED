#!/bin/bash
# SAM3 전환 가능 여부 확인 (설치/다운로드 없이 점검만)
# 실행: bash tools/check_sam3_env.sh

echo "=== CUDA (nvcc) ==="
nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "nvcc 없음"

echo ""
echo "=== GPU 드라이버 (nvidia-smi) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi 없음"

echo ""
echo "=== Python 3.12 사용 가능 여부 ==="
for cmd in python3.12 python3.13; do
    if command -v $cmd &>/dev/null; then
        echo "$cmd 있음: $($cmd --version)"
    else
        echo "$cmd 없음"
    fi
done
echo "conda로 설치 가능한 python 버전은 'conda search python' 참고"

echo ""
echo "=== HuggingFace CLI / 로그인 상태 ==="
if command -v hf &>/dev/null; then
    echo "hf CLI 있음"
    hf auth whoami 2>&1
elif command -v huggingface-cli &>/dev/null; then
    echo "huggingface-cli 있음 (구버전)"
    huggingface-cli whoami 2>&1
else
    echo "hf/huggingface-cli 없음 (pip install huggingface_hub 필요)"
fi

echo ""
echo "=== SAM3 체크포인트 접근권 신청 필요 (직접 확인) ==="
echo "1) https://huggingface.co/facebook/sam3.1 에서 접근 신청/승인 여부 확인"
echo "2) 승인됐으면: hf auth login 으로 토큰 등록"
echo ""
echo "위 CUDA/Python 결과를 보고 SAM3(요구사항: Python 3.12+, PyTorch 2.7+, CUDA 12.6+) 설치 가능한지 판단"
