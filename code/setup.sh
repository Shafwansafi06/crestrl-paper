#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$DIR/workdir"

echo "============================================================"
echo "  AnchorGRPO Setup"
echo "============================================================"
echo ""

# GPU check
echo "[1/6] GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Auto-detect CUDA version for PyTorch
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
if [ "$CUDA_MAJOR" -ge 535 ]; then
    PYTORCH_CUDA="cu121"
elif [ "$CUDA_MAJOR" -ge 520 ]; then
    PYTORCH_CUDA="cu118"
else
    PYTORCH_CUDA="cu118"
fi
echo "  Driver: $CUDA_VER → PyTorch: $PYTORCH_CUDA"
echo ""

# System deps
echo "[2/6] System packages..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-pip python3-venv git wget
fi

# Python venv
echo "[3/6] Python venv..."
python3 -m venv "$WORK/.venv"
source "$WORK/.venv/bin/activate"
pip install --upgrade pip setuptools wheel -q

# PyTorch + CUDA
echo "[4/6] PyTorch + $PYTORCH_CUDA..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA" -q
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_mem', getattr(props, 'total_memory', 0))
    print(f'  {torch.cuda.get_device_name(0)} — {mem/1024**3:.0f}GB')
"

# ML packages
echo "[5/6] ML dependencies..."
pip install -q "transformers>=4.40.0" "peft>=0.10.0" "trl>=0.12.0" "datasets>=2.18.0"
pip install -q "accelerate>=0.28.0" "bitsandbytes>=0.43.0" sentencepiece protobuf scipy
pip install -q scikit-learn numpy matplotlib seaborn pandas tabulate
pip install -q huggingface_hub safetensors rank_bm25 sentence_transformers

# Download model
echo "[6/6] Downloading Mistral-7B-Instruct-v0.3 (~14GB)..."
python3 -c "
from huggingface_hub import snapshot_download
import os
d = os.path.join('$WORK', 'model_cache', 'Mistral-7B-Instruct-v0.3')
os.makedirs(d, exist_ok=True)
snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', local_dir=d, local_dir_use_symlinks=False)
print('  Done.')
"

echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  source workdir/.venv/bin/activate"
echo "  bash run_all.sh"
echo ""
