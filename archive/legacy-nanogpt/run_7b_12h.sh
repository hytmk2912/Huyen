#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
WORKDIR=${WORKDIR:-$HOME/ai-agent}
mkdir -p "$WORKDIR"
cd "$WORKDIR"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git tmux
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy tiktoken datasets huggingface_hub tqdm
mkdir -p nanoGPT && cd nanoGPT
curl -fsSL https://raw.githubusercontent.com/hytmk2912/Huyen/main/model_7b_rope.py -o model_7b_rope.py
curl -fsSL https://raw.githubusercontent.com/hytmk2912/Huyen/main/prepare_reason.py -o prepare_reason.py
curl -fsSL https://raw.githubusercontent.com/hytmk2912/Huyen/main/train_7b_65536.py -o train_7b_65536.py
export OUT_DIR=data/reason_7b TARGET_TOKENS=80000000
if [[ ! -f data/reason_7b/train.bin ]]; then python prepare_reason.py; fi
export DATA_DIR=data/reason_7b CKPT_DIR=out-7b-65536 BLOCK_SIZE=65536 TARGET_HOURS=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_7b_65536.py
