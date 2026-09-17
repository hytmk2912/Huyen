#!/usr/bin/env bash
# VPS GPU MOI — 1 lenh. Bat buoc:
#   export CPU_IP=x.x.x.x CPU_USER=root HF_TOKEN=hf_xxx
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
CPU_IP="${CPU_IP:-}"
CPU_USER="${CPU_USER:-root}"
CPU_PORT="${CPU_PORT:-22}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-hytmk2912/huyen-code-7b}"
TARGET_HOURS="${TARGET_HOURS:-7.5}"
WORKDIR="${WORKDIR:-/root/ai-agent}"
DATA_DIR="$WORKDIR/nanoGPT/data/code_7b"
CKPT_DIR="$WORKDIR/nanoGPT/out-code-7b"
OFFLOAD="$WORKDIR/nvme_offload"
REMOTE="${CPU_USER}@${CPU_IP}"
RSYNC_SSH="ssh -p ${CPU_PORT} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
echo "== [GPU] Kiem tra bien =="
[[ -n "$CPU_IP" ]] || { echo THIEU CPU_IP; exit 1; }
[[ -n "$HF_TOKEN" ]] || { echo THIEU HF_TOKEN — tao https://huggingface.co/settings/tokens; exit 1; }
echo "== [GPU] Apt + swap 96G =="
apt-get update -y
apt-get install -y python3 python3-pip python3-venv git rsync openssh-client build-essential python3-dev libaio-dev pigz zstd curl ca-certificates
if [[ ! -f /swapfile_train ]]; then
  fallocate -l 96G /swapfile_train || dd if=/dev/zero of=/swapfile_train bs=1M count=98304
  chmod 600 /swapfile_train; mkswap /swapfile_train
fi
swapon /swapfile_train || true
sysctl -w vm.swappiness=60 || true
echo "== [GPU] venv + torch CUDA =="
mkdir -p "$WORKDIR"; cd "$WORKDIR"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu128 || pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy transformers datasets tiktoken tqdm deepspeed huggingface_hub safetensors
[[ -d $WORKDIR/nanoGPT/.git ]] || git clone https://github.com/karpathy/nanoGPT.git "$WORKDIR/nanoGPT"
cd "$WORKDIR/nanoGPT"
mkdir -p "$DATA_DIR" "$CKPT_DIR" "$OFFLOAD" /root/.ssh
[[ -f /root/.ssh/id_ed25519 ]] || ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519
echo "== [GPU] PUBLIC KEY — dan vao authorized_keys VPS CPU =="
cat /root/.ssh/id_ed25519.pub
ok=0
for i in $(seq 1 30); do
  if $RSYNC_SSH -o ConnectTimeout=8 -o BatchMode=yes "$REMOTE" "echo ssh_ok" >/dev/null 2>&1; then ok=1; break; fi
  echo "chua SSH duoc ($i/30)"; sleep 10
done
[[ "$ok" == 1 ]] || { echo KHONG SSH duoc $REMOTE; exit 1; }
$RSYNC_SSH "$REMOTE" "mkdir -p ~/ai-agent/nanoGPT/data/code_7b ~/ai-agent/nanoGPT/out-code-7b"
rsync -avP -e "$RSYNC_SSH" "$REMOTE:~/ai-agent/nanoGPT/data/code_7b/" "$DATA_DIR/" || true
if [[ ! -f $DATA_DIR/train.bin ]]; then
  rsync -avP -e "$RSYNC_SSH" "$REMOTE:~/ai-agent/nanoGPT/data/code_3b/train.bin" "$DATA_DIR/" || true
  rsync -avP -e "$RSYNC_SSH" "$REMOTE:~/ai-agent/nanoGPT/data/code_3b/val.bin" "$DATA_DIR/" || true
fi
rsync -avP -e "$RSYNC_SSH" "$REMOTE:~/ai-agent/nanoGPT/out-code-7b/" "$CKPT_DIR/" || true
[[ -f $DATA_DIR/train.bin ]] || { echo KHONG CO train.bin; exit 1; }
python3 - << PY
import os
p="$DATA_DIR/train.bin"
print(f"train.bin = {os.path.getsize(p)/1e9:.3f} GB (~{os.path.getsize(p)//2:,} token)")
PY
curl -fsSL https://raw.githubusercontent.com/hytmk2912/Huyen/main/ds_config_7b.json -o ds_config_7b.json
cat > train_7b.py << 'PY'
import os, time, glob, shutil
import numpy as np, torch, deepspeed
from model import GPTConfig, GPT
DATA_DIR = os.environ.get("DATA_DIR", "data/code_7b")
CKPT_DIR = os.environ.get("CKPT_DIR", "out-code-7b")
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "1024"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "1"))
GRAD_ACCUM = 32
TARGET_HOURS = float(os.environ.get("TARGET_HOURS", "7.5"))
LOG_EVERY = 5
SAVE_EVERY_SEC = 900
KEEP = 2
def get_batch(split):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    if split == "val" and not os.path.exists(path):
        path = os.path.join(DATA_DIR, "train.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(data) - BLOCK_SIZE
    if n <= 0: raise SystemExit("train.bin qua ngan")
    ix = torch.randint(n, (MICRO_BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x, y
def rotate_ckpts():
    tags = sorted([p for p in glob.glob(os.path.join(CKPT_DIR, "step_*")) if os.path.isdir(p)], key=os.path.getmtime)
    while len(tags) > KEEP:
        old = tags.pop(0); shutil.rmtree(old, ignore_errors=True); print("Xoa", old)
os.makedirs(CKPT_DIR, exist_ok=True)
train_size = os.path.getsize(os.path.join(DATA_DIR, "train.bin")) // 2
tokens_per_step = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
print(f"Train tokens: {train_size:,}. tokens/step {tokens_per_step:,}")
config = GPTConfig(block_size=BLOCK_SIZE, vocab_size=50304, n_layer=32, n_head=32, n_embd=4096, dropout=0.0, bias=False)
n_params = sum(p.numel() for p in GPT(config).parameters())
print(f"GPT 7B-class: {n_params/1e9:.2f}B (khong phai Llama)")
model = GPT(config)
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config="ds_config_7b.json")
it = tokens_seen = 0
try:
    load_path, client_state = model_engine.load_checkpoint(CKPT_DIR, tag="latest")
except Exception as e:
    load_path, client_state = None, {}
    print("Khong load latest:", e)
if load_path is not None:
    it = int(client_state.get("it", 0)); tokens_seen = int(client_state.get("tokens_seen", 0))
    print(f"RESUME iter={it:,} seen={tokens_seen:,}")
else:
    print("Train tu dau")
start = last_save = time.time(); calibrated = False
os.makedirs("/root/ai-agent/nvme_offload", exist_ok=True)
while True:
    if (time.time() - start) / 3600 >= TARGET_HOURS:
        print(f"Dat {TARGET_HOURS}h — dung de copy ve CPU"); break
    x, y = get_batch("train")
    x, y = x.to(model_engine.device), y.to(model_engine.device)
    _, loss = model_engine(x, y)
    model_engine.backward(loss); model_engine.step()
    it += 1; tokens_seen += tokens_per_step
    if it == 5 and not calibrated:
        sec = (time.time() - start) / 5
        print(f"[TOC DO] ~{sec:.1f}s/step ~{tokens_per_step/sec:.0f} tok/s"); calibrated = True
    if it % LOG_EVERY == 0:
        print(f"iter {it} ({(time.time()-start)/3600:.2f}h) loss={loss.item():.4f} seen={tokens_seen:,}", flush=True)
    if time.time() - last_save >= SAVE_EVERY_SEC:
        model_engine.save_checkpoint(CKPT_DIR, tag=f"step_{it}", client_state={"it": it, "tokens_seen": tokens_seen})
        model_engine.save_checkpoint(CKPT_DIR, tag="latest", client_state={"it": it, "tokens_seen": tokens_seen})
        rotate_ckpts(); last_save = time.time()
model_engine.save_checkpoint(CKPT_DIR, tag=f"step_{it}", client_state={"it": it, "tokens_seen": tokens_seen})
model_engine.save_checkpoint(CKPT_DIR, tag="latest", client_state={"it": it, "tokens_seen": tokens_seen})
rotate_ckpts()
print(f"XONG train iter={it:,} seen={tokens_seen:,}")
PY
echo "== [GPU] Train ${TARGET_HOURS}h =="
export DATA_DIR CKPT_DIR TARGET_HOURS
set +e
deepspeed train_7b.py; rc=$?
if [[ $rc -ne 0 ]]; then export BLOCK_SIZE=512; deepspeed train_7b.py; rc=$?; fi
set -e
CONVERT=$(find "$CKPT_DIR" "$WORKDIR/nanoGPT" -name zero_to_fp32.py 2>/dev/null | head -n1 || true)
[[ -n "$CONVERT" ]] && python3 "$CONVERT" "$CKPT_DIR" "$WORKDIR/nanoGPT/model_fp32.pt" || true
python3 - << 'PY'
import os, glob, shutil
ck=os.environ.get("CKPT_DIR","/root/ai-agent/nanoGPT/out-code-7b")
tags=sorted([p for p in glob.glob(os.path.join(ck,"step_*")) if os.path.isdir(p)], key=os.path.getmtime)
for p in tags[:-2]:
    shutil.rmtree(p, ignore_errors=True); print("xoa", p)
print("giu", tags[-2:])
PY
python3 - << PY
import os
from huggingface_hub import HfApi, login
login(token=os.environ["HF_TOKEN"])
api = HfApi(); repo = os.environ.get("HF_REPO", "hytmk2912/huyen-code-7b")
api.create_repo(repo, repo_type="model", exist_ok=True, private=True)
pt = "/root/ai-agent/nanoGPT/model_fp32.pt"
if os.path.exists(pt):
    api.upload_file(path_or_fileobj=pt, path_in_repo="model_fp32.pt", repo_id=repo, repo_type="model")
    print("uploaded model_fp32.pt")
PY
$RSYNC_SSH "$REMOTE" "mkdir -p ~/ai-agent/nanoGPT/out-code-7b ~/ai-agent/nanoGPT"
rsync -avP --delete -e "$RSYNC_SSH" "$CKPT_DIR/" "$REMOTE:~/ai-agent/nanoGPT/out-code-7b/" || true
[[ -f $WORKDIR/nanoGPT/model_fp32.pt ]] && rsync -avP -e "$RSYNC_SSH" "$WORKDIR/nanoGPT/model_fp32.pt" "$REMOTE:~/ai-agent/nanoGPT/" || true
echo "== [GPU] XONG. Checkpoint o VPS CPU + HF $HF_REPO =="
