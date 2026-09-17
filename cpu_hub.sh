#!/usr/bin/env bash
# VPS CU (khong GPU) — dan 1 lenh. May nay GIU data + checkpoint.
# Tuy chon:
#   export GPU_PUBKEY="ssh-ed25519 AAAA... root@gpu"
#   export PREPARE_MORE=1
#   export TARGET_MORE_TOKENS=200000000
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

PREPARE_MORE="${PREPARE_MORE:-1}"
TARGET_MORE_TOKENS="${TARGET_MORE_TOKENS:-200000000}"
WORKDIR="${WORKDIR:-$HOME/ai-agent}"
DATA3="$WORKDIR/nanoGPT/data/code_3b"
DATA7="$WORKDIR/nanoGPT/data/code_7b"
NEXT="$WORKDIR/nanoGPT/data/next_shard"
CKPT="$WORKDIR/nanoGPT/out-code-7b"

echo "== [CPU] Cai goi =="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git rsync openssh-server curl ca-certificates
sudo systemctl enable --now ssh || sudo service ssh start || true
mkdir -p "$DATA3" "$DATA7" "$NEXT" "$CKPT" "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"

if [[ -n "${GPU_PUBKEY:-}" ]]; then
  grep -qxF "$GPU_PUBKEY" "$HOME/.ssh/authorized_keys" || echo "$GPU_PUBKEY" >> "$HOME/.ssh/authorized_keys"
  echo "Da them GPU_PUBKEY vao authorized_keys"
fi

echo "== [CPU] authorized_keys + IP =="
cat "$HOME/.ssh/authorized_keys" || true
hostname -I || true
curl -s ifconfig.me || true
echo

if [[ -f "$DATA3/train.bin" && ! -f "$DATA7/train.bin" ]]; then
  ln -f "$DATA3/train.bin" "$DATA7/train.bin" 2>/dev/null || cp -a "$DATA3/train.bin" "$DATA7/train.bin"
  if [[ -f "$DATA3/val.bin" ]]; then
    ln -f "$DATA3/val.bin" "$DATA7/val.bin" 2>/dev/null || cp -a "$DATA3/val.bin" "$DATA7/val.bin"
  fi
fi

if [[ -f "$DATA7/train.bin" ]]; then
  python3 - << PY
import os
p="$DATA7/train.bin"
print(f"[CPU] train.bin: {os.path.getsize(p)/1e9:.3f} GB (~{os.path.getsize(p)//2:,} token)")
PY
else
  echo "CHUA CO train.bin. Chay step1_prepare_data.sh neu can build data."
fi

echo "== [CPU] Lang nghe SSH/rsync. GIU terminal nay mo."

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
print(f"Tai them toi da {target:,} token vao {out_dir}")
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
    try:
        apps = load_dataset("codeparrot/apps", split="train")
        for x in apps:
            if total >= target: break
            try:
                sols = json.loads(x["solutions"]) if x["solutions"] else []
            except Exception:
                sols = []
            if sols:
                write(tr, va, f"Problem ({x['difficulty']}): {x['question']}\nSolution:\n{sols[0]}\n")
    except Exception as e:
        print("apps:", e)
print(f"XONG shard moi: {total:,} token -> {train_p}")
PY
  NEXT_DIR="$NEXT" TARGET_MORE_TOKENS="$TARGET_MORE_TOKENS" python3 "$WORKDIR/nanoGPT/prepare_next_shard.py" > "$WORKDIR/prepare_next.log" 2>&1 &
  echo "PID tai them data: $! log=$WORKDIR/prepare_next.log"
fi

last=0
while true; do
  if [[ -d "$CKPT" ]]; then
    sz=$(du -sb "$CKPT" 2>/dev/null | awk '{print $1}')
    if [[ "${sz:-0}" -ne "$last" ]]; then
      echo "[$(date -Is)] checkpoint dir = $((sz/1024/1024)) MB"
      last=$sz
    fi
  fi
  sleep 600
done
