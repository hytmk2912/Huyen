#!/usr/bin/env bash
# VPS CPU — 1 lenh. GIU terminal mo.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
PREPARE_MORE="${PREPARE_MORE:-1}"
TARGET_MORE_TOKENS="${TARGET_MORE_TOKENS:-200000000}"
SYNC_PASS="${SYNC_PASS:-huyen-sync-7b}"
WORKDIR="${WORKDIR:-$HOME/ai-agent}"
DATA3="$WORKDIR/nanoGPT/data/code_3b"
DATA7="$WORKDIR/nanoGPT/data/code_7b"
NEXT="$WORKDIR/nanoGPT/data/next_shard"
CKPT="$WORKDIR/nanoGPT/out-code-7b"

sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git rsync curl ca-certificates
mkdir -p "$DATA3" "$DATA7" "$NEXT" "$CKPT" "$WORKDIR/incoming"

if [[ -f "$DATA3/train.bin" && ! -f "$DATA7/train.bin" ]]; then
  ln -f "$DATA3/train.bin" "$DATA7/train.bin" 2>/dev/null || cp -a "$DATA3/train.bin" "$DATA7/train.bin"
  [[ -f "$DATA3/val.bin" ]] && { ln -f "$DATA3/val.bin" "$DATA7/val.bin" 2>/dev/null || cp -a "$DATA3/val.bin" "$DATA7/val.bin"; }
fi

# rsync daemon — GPU keo/day file khong can SSH key
SECRETS=/tmp/huyen.rsync.secrets
CONF=/tmp/huyen.rsyncd.conf
printf 'huyen:%s\n' "$SYNC_PASS" | sudo tee "$SECRETS" >/dev/null
sudo chmod 600 "$SECRETS"
cat > "$CONF" << EOF
pid file = /tmp/huyen-rsyncd.pid
lock file = /tmp/huyen-rsyncd.lock
log file = /tmp/huyen-rsyncd.log
port = 8730
use chroot = no
[huyen]
path = $WORKDIR
comment = huyen train hub
read only = false
list = yes
auth users = huyen
secrets file = $SECRETS
uid = $(id -u)
gid = $(id -g)
EOF
sudo pkill -f 'rsync --daemon --config=/tmp/huyen.rsyncd.conf' 2>/dev/null || true
sudo rsync --daemon --config="$CONF"

IP=$(curl -s --max-time 8 ifconfig.me || hostname -I | awk '{print $1}')
echo "========================================"
echo "VPS CPU SAN SANG"
echo "IP public: $IP"
echo "rsync: rsync://huyen@$IP:8730/huyen/"
echo "Dien IP nay vao CPU_IP o lenh GPU"
echo "========================================"
if [[ -f "$DATA7/train.bin" ]]; then
  python3 - << PY
import os
p="$DATA7/train.bin"
print(f"train.bin: {os.path.getsize(p)/1e9:.3f} GB (~{os.path.getsize(p)//2:,} token)")
PY
else
  echo "CHUA CO train.bin tai $DATA7 — GPU se khong train duoc."
fi

if [[ "$PREPARE_MORE" == "1" ]]; then
  pip3 install --break-system-packages numpy tiktoken datasets tqdm || true
  mkdir -p "$WORKDIR/nanoGPT"
  cat > "$WORKDIR/nanoGPT/prepare_next_shard.py" << 'PY'
import os, json
import numpy as np, tiktoken
from datasets import load_dataset
out_dir = os.path.expanduser(os.environ.get("NEXT_DIR", "~/ai-agent/nanoGPT/data/next_shard"))
os.makedirs(out_dir, exist_ok=True)
target = int(os.environ.get("TARGET_MORE_TOKENS", "200000000"))
enc = tiktoken.get_encoding("gpt2")
train_p = os.path.join(out_dir, "train.bin")
val_p = os.path.join(out_dir, "val.bin")
total = 0
def write(tr, va, text):
    global total
    if not text: return
    ids = enc.encode_ordinary(text)
    if not ids: return
    arr = np.array(ids, dtype=np.uint16)
    (tr if np.random.random() < 0.98 else va).write(arr.tobytes())
    total += len(ids)
print(f"Tai them toi da {target:,} token")
with open(train_p, "ab") as tr, open(val_p, "ab") as va:
    try:
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        for i, x in enumerate(fw):
            if total >= int(target * 0.6): break
            write(tr, va, x.get("text", ""))
            if i % 1000 == 0: print(f"fineweb ~{total:,}", flush=True)
    except Exception as e:
        print("fineweb:", e)
    try:
        st = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for x in st:
            if total >= int(target * 0.75): break
            write(tr, va, x.get("text", ""))
    except Exception as e:
        print("tinystories:", e)
    try:
        gs = load_dataset("openai/gsm8k", "main", split="train")
        for x in gs:
            write(tr, va, f"Question: {x['question']}\nAnswer: {x['answer']}\n")
    except Exception as e:
        print("gsm8k:", e)
print(f"XONG shard moi: {total:,} token")
PY
  NEXT_DIR="$NEXT" TARGET_MORE_TOKENS="$TARGET_MORE_TOKENS" python3 "$WORKDIR/nanoGPT/prepare_next_shard.py" > "$WORKDIR/prepare_next.log" 2>&1 &
  echo "Dang tai them data PID=$! log=$WORKDIR/prepare_next.log"
fi

echo "GIU terminal nay. Mo firewall TCP 8730 neu nha cung cap chan."
last=0
while true; do
  sz=$(du -sb "$CKPT" 2>/dev/null | awk '{print $1}')
  if [[ "${sz:-0}" -ne "$last" ]]; then
    echo "[$(date -Is)] checkpoint = $((sz/1024/1024)) MB"
    last=$sz
  fi
  sleep 300
done
