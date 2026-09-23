#!/usr/bin/env bash
set -e
cd ~/ai-agent/nanoGPT

cat > data/code_3b/step1b.py << 'EOF'
import os
import shutil
import numpy as np
import tiktoken
from datasets import load_dataset

DATA_DIR = os.path.dirname(__file__)
KEEP_FREE_GB = 5  # de lai it nhat 5GB cho he thong, khong dung het 100% dia

free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
usable_gb = max(0, free_gb - KEEP_FREE_GB)
target_new_tokens = int(usable_gb * 1e9 / 2)  # 2 byte/token (uint16)
print(f"Con {free_gb:.0f}GB trong dia, se dung them ~{usable_gb:.0f}GB -> muc tieu them ~{target_new_tokens:,} token moi.")
if target_new_tokens < 1_000_000:
    raise SystemExit("Khong con du dia de them du lieu moi co y nghia - dung lai.")

enc = tiktoken.get_encoding("gpt2")
new_tokens = 0

def write_chunk(train_fh, val_fh, text):
    global new_tokens
    if not text:
        return
    ids = enc.encode_ordinary(text)
    if not ids:
        return
    arr = np.array(ids, dtype=np.uint16)
    if np.random.random() < 0.98:
        arr.tofile(train_fh)
    else:
        arr.tofile(val_fh)
    new_tokens += len(ids)

# Mo bang che do "ab" (append) - noi tiep vao file cu, KHONG ghi de/xoa du lieu da co.
with open(os.path.join(DATA_DIR, "train.bin"), "ab") as train_fh, \
     open(os.path.join(DATA_DIR, "val.bin"), "ab") as val_fh:

    # FineWeb-Edu ban 100 ty token (thay vi 10 ty da dung o Buoc 1) - khong bi khoa,
    # thua suc lap day phan dia con lai, co the trung 1 phan nho voi ban 10 ty cu (chap nhan duoc).
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
    for x in fw:
        if new_tokens >= target_new_tokens:
            break
        write_chunk(train_fh, val_fh, x.get("text", ""))
        if new_tokens % 500_000_000 < 2000:
            print(f"Da them: ~{new_tokens:,} token moi")

print(f"XONG BUOC 1b: them ~{new_tokens:,} token moi vao train.bin/val.bin (cong don voi du lieu Buoc 1 cu).")

import datetime
with open(os.path.join(DATA_DIR, "DATA_MANIFEST.txt"), "a") as mf:
    mf.write(f"[Buoc 1b - {datetime.datetime.now().isoformat()}]\n")
    mf.write(f"Them: {new_tokens:,} token tu HuggingFaceFW/fineweb-edu (sample-100BT)\n\n")
print("Da cap nhat DATA_MANIFEST.txt.")

try:
    import urllib.request as _ur
    _ur.urlopen(_ur.Request(
        "https://ntfy.sh/huyenb2404-vps-databuild-x7q2",
        data=f"XONG Buoc 1b: them {new_tokens:,} token moi (FineWeb-Edu 100BT). Co the chuyen sang Buoc 2.".encode(),
        method="POST",
    ), timeout=10)
except Exception as e:
    print(f"Khong gui duoc thong bao (khong sao): {e}")
EOF
python3 data/code_3b/step1b.py
