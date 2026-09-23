#!/usr/bin/env bash
set -e

echo "== Cai Python + thu vien =="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git
pip3 install --break-system-packages torch numpy transformers datasets tiktoken tqdm deepspeed

echo "== Tai nanoGPT =="
mkdir -p ~/ai-agent
git clone https://github.com/karpathy/nanoGPT.git ~/ai-agent/nanoGPT
cd ~/ai-agent/nanoGPT
mkdir -p data/code_3b

cat > data/code_3b/prepare.py << 'EOF'
import os
import subprocess
import shutil
import time
import numpy as np
import tiktoken
from datasets import load_dataset

TARGET_TOKENS = 15_000_000_000  # dung het 75GB dia theo yeu cau - xem tinh toan trong README
# Chi phi CO DINH tren dia, khong doi theo TARGET_TOKENS:
# ~30 repo code (depth 1, ~5GB) + 12 repo lay lich su (depth 300, nang hon, ~12GB) + cache Hugging Face (~10GB)
FIXED_OVERHEAD_GB = 5 + 12 + 10
# Thiet ke ghi thang vao train.bin/val.bin (khong qua file gop trung gian nua) ->
# dung luong token cuoi cung = TARGET_TOKENS * 2 byte (uint16), khong con canh "dinh gap doi" luc chia truoc day.
token_data_gb = (TARGET_TOKENS * 2) / 1e9
need_gb = token_data_gb + FIXED_OVERHEAD_GB
free_gb = shutil.disk_usage(os.path.dirname(__file__)).free / 1e9
print(f"Uoc tinh Buoc 1 can ~{need_gb:.0f}GB (token data ~{token_data_gb:.0f}GB + repo/cache co dinh ~{FIXED_OVERHEAD_GB}GB), "
      f"con {free_gb:.0f}GB trong.")
print("Luu y: neu may TRAIN (co GPU) la may khac may nay, may do can rieng ~35GB nua "
      "(checkpoint FP32 + du phong NVMe offload) - kiem tra o may do lúc chuan bi train.")
if free_gb < need_gb * 1.15:
    raise SystemExit(
        f"KHONG DU DIA: can khoang {need_gb:.0f}GB (co du phong), chi con {free_gb:.0f}GB. "
        f"Giam TARGET_TOKENS trong file nay xuong roi chay lai, hoac them dia cho VPS."
    )

enc = tiktoken.get_encoding("gpt2")
assert enc.n_vocab <= 50304, "Tokenizer co vocab lon hon vocab_size cua model (50304) - bao AI biet"

def clone_with_retry(url, dest, depth, tries=3):
    for i in range(tries):
        try:
            subprocess.run(["git", "clone", "--depth", str(depth), url, dest], check=True, timeout=600)
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Loi clone {url} (lan {i+1}/{tries}): {e}")
            if os.path.exists(dest):
                subprocess.run(["rm", "-rf", dest])
            time.sleep(10 * (i + 1))
    return False

total_tokens = 0
train_path = os.path.join(os.path.dirname(__file__), "train.bin")
val_path = os.path.join(os.path.dirname(__file__), "val.bin")

def write_chunk(train_fh, val_fh, text):
    # Chia train/val ngay luc ghi (~98/2), khong qua file gop trung gian ->
    # tranh can gap doi dung luong dia cung luc, quan trong khi dung gan het 75GB.
    global total_tokens
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
    total_tokens += len(ids)

REPOS = [
    "flask:pallets/flask", "requests:psf/requests", "click:pallets/click",
    "rich:Textualize/rich", "httpx:encode/httpx", "tqdm:tqdm/tqdm",
    "pytest:pytest-dev/pytest", "django:django/django", "numpy:numpy/numpy",
    "pandas:pandas-dev/pandas", "scikit-learn:scikit-learn/scikit-learn",
    "matplotlib:matplotlib/matplotlib", "sqlalchemy:sqlalchemy/sqlalchemy",
    "pydantic:pydantic/pydantic", "fastapi:tiangolo/fastapi",
    "starlette:encode/starlette", "aiohttp:aio-libs/aiohttp",
    "celery:celery/celery", "scrapy:scrapy/scrapy", "pillow:python-pillow/Pillow",
    "black:psf/black", "mypy:python/mypy", "sympy:sympy/sympy",
    "networkx:networkx/networkx", "streamlit:streamlit/streamlit",
    "typer:tiangolo/typer", "pyyaml:yaml/pyyaml", "cryptography:pyca/cryptography",
    "paramiko:paramiko/paramiko", "gunicorn:benoitc/gunicorn",
]
work_dir = os.path.join(os.path.dirname(__file__), "_src")
os.makedirs(work_dir, exist_ok=True)

with open(train_path, "wb") as train_fh, open(val_path, "wb") as val_fh:
    # 1) Code that - clone ~30 repo GitHub mo
    for entry in REPOS:
        if total_tokens >= TARGET_TOKENS:
            break
        name, gh = entry.split(":")
        dest = os.path.join(work_dir, name)
        if not os.path.exists(dest):
            if not clone_with_retry(f"https://github.com/{gh}.git", dest, depth=1):
                print(f"Bo qua {gh} sau 3 lan thu")
                continue
        for root, _, files in os.walk(dest):
            for fn in files:
                if fn.endswith(".py"):
                    try:
                        with open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore") as f:
                            write_chunk(train_fh, val_fh, f.read())
                    except Exception:
                        pass
        print(f"Sau {name}: ~{total_tokens:,} token")

    # 2) Van ban chat luong cao quy mo lon (mau co san 10 ty token, lay den khi het mau hoac du ty le)
    if total_tokens < TARGET_TOKENS * 0.55:
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        for x in fw:
            if total_tokens >= TARGET_TOKENS * 0.55:
                break
            write_chunk(train_fh, val_fh, x.get("text", ""))
        print(f"Sau FineWeb-Edu: ~{total_tokens:,} token")

    # 3) Wikipedia Anh + Viet, TinyStories - KHONG gioi han so bai nho nhu truoc (muc tieu lon hon
    # nhieu lan roi), de moi nguon tu chay den khi het du lieu that hoac du ty le muc tieu
    extra_sources = [
        ("wikimedia/wikipedia", "20231101.en", 0.85),
        ("wikimedia/wikipedia", "20231101.vi", 0.90),
        ("roneneldan/TinyStories", None, 1.0),
    ]
    for ds_name, cfg, stop_ratio in extra_sources:
        if total_tokens >= TARGET_TOKENS:
            break
        stream = load_dataset(ds_name, cfg, split="train", streaming=True) if cfg else \
                 load_dataset(ds_name, split="train", streaming=True)
        for x in stream:
            if total_tokens >= TARGET_TOKENS * stop_ratio:
                break
            write_chunk(train_fh, val_fh, x.get("text", ""))
        print(f"Sau {ds_name} {cfg or ''}: ~{total_tokens:,} token")

    # 4) Tu duy tinh toan + suy luan khoa hoc
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    for x in math_ds:
        write_chunk(train_fh, val_fh, f"Question: {x['question']}\nAnswer: {x['answer']}\n")

    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    for x in arc_ds:
        pairs = list(zip(x["choices"]["label"], x["choices"]["text"]))
        choices_str = "\n".join(f"{lbl}) {txt}" for lbl, txt in pairs)
        correct = dict(pairs).get(x["answerKey"], "")
        write_chunk(train_fh, val_fh, f"Question: {x['question']}\nChoices:\n{choices_str}\nAnswer: {x['answerKey']}) {correct}\n")

    # 5) Toan kho hon, giai chi tiet tung buoc (bo sung GSM8K de tu duy sau hon)
    math_hard = load_dataset("hendrycks/competition_math", split="train")
    for x in math_hard:
        write_chunk(train_fh, val_fh, f"Problem: {x['problem']}\nSolution: {x['solution']}\n")

    # 6) Tu duy "bat loi + sua loi" that - lay tu chinh cac commit sua bug trong code that
    HISTORY_REPOS = [
        "pallets/click", "psf/requests", "pallets/flask", "tqdm/tqdm", "encode/httpx",
        "pytest-dev/pytest", "pandas-dev/pandas", "django/django", "numpy/numpy",
        "scikit-learn/scikit-learn", "pydantic/pydantic", "psf/black",
    ]
    for gh in HISTORY_REPOS:
        name = gh.split("/")[-1]
        dest = os.path.join(work_dir, f"{name}_hist")
        if not os.path.exists(dest):
            if not clone_with_retry(f"https://github.com/{gh}.git", dest, depth=300):
                print(f"Bo qua lich su {gh} sau 3 lan thu")
                continue
        try:
            log = subprocess.run(
                ["git", "-C", dest, "log", "--oneline", "-40", "--grep=fix", "-i"],
                capture_output=True, text=True, timeout=30
            ).stdout.strip().splitlines()
        except Exception:
            continue
        for line in log:
            h = line.split()[0]
            try:
                diff = subprocess.run(
                    ["git", "-C", dest, "show", h, "-p"],
                    capture_output=True, text=True, timeout=30
                ).stdout
                if 200 < len(diff) < 8000:
                    write_chunk(train_fh, val_fh, f"# Vi du sua loi that trong code (commit that tu {gh}):\n{diff}\n")
            except Exception:
                continue

    # 7) Kien thuc trading/thi truong - KHAI NIEM thoi, CHUA phai du lieu nen/gia
    import urllib.request
    import urllib.parse
    import json as jsonlib

    TRADING_TOPICS = [
        "Financial market", "Stock exchange", "Order (exchange)", "Market maker",
        "Bid-ask spread", "Limit order", "Market order", "Short (finance)",
        "Leverage (finance)", "Margin (finance)", "Volatility (finance)",
        "Technical analysis", "Candlestick chart", "Support and resistance (technical analysis)",
        "Moving average", "Relative strength index", "Bollinger Bands", "MACD",
        "Fibonacci retracement", "Trading volume", "Market capitalization", "Market liquidity",
        "Order book", "Futures contract", "Option (finance)", "Derivative (finance)",
        "Risk management", "Diversification (finance)", "Cryptocurrency", "Bitcoin",
        "Ethereum", "Blockchain", "Decentralized finance", "Stablecoin",
        "Initial coin offering", "Market sentiment", "Fundamental analysis",
        "Day trading", "Swing trading", "Arbitrage",
    ]
    for title in TRADING_TOPICS:
        success = False
        for attempt in range(2):
            try:
                q = urllib.parse.urlencode({
                    "action": "query", "format": "json", "prop": "extracts",
                    "explaintext": "true", "redirects": "1", "titles": title,
                })
                req = urllib.request.Request(
                    f"https://en.wikipedia.org/w/api.php?{q}",
                    headers={"User-Agent": "personal-research-project/1.0"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = jsonlib.loads(resp.read().decode("utf-8"))
                for page in data.get("query", {}).get("pages", {}).values():
                    text = page.get("extract", "")
                    if text:
                        write_chunk(train_fh, val_fh, f"# {page.get('title', title)}\n{text}\n")
                success = True
                break
            except Exception as e:
                print(f"Loi bai '{title}' (lan {attempt+1}/2): {e}")
                time.sleep(5 * (attempt + 1))
        time.sleep(0.5)
        if not success:
            print(f"Bo qua han bai '{title}'")

    # 8) Hoi-dap dau tu/tai chinh tong quat (khai niem, khong phai loi khuyen dau tu that)
    fin_qa = load_dataset("gbharti/finance-alpaca", split="train")
    for x in fin_qa:
        instr = x.get("instruction", "") or ""
        inp = x.get("input", "") or ""
        out = x.get("output", "") or ""
        write_chunk(train_fh, val_fh, f"Question: {instr}" + (f"\n{inp}" if inp else "") + f"\nAnswer: {out}\n")

    # 9) Giai bai toan lap trinh that (co do kho), de hoc tu duy giai quyet van de bang code
    apps_ds = load_dataset("codeparrot/apps", split="train")
    for x in apps_ds:
        try:
            sols = jsonlib.loads(x["solutions"]) if x["solutions"] else []
        except Exception:
            sols = []
        if not sols:
            continue
        write_chunk(train_fh, val_fh, f"Problem ({x['difficulty']}): {x['question']}\nSolution:\n{sols[0]}\n")

print(f"TONG: ~{total_tokens:,} token thu duoc (muc tieu: {TARGET_TOKENS:,}).")

# Don dep repo da clone - khong can nua sau khi da tokenize xong, giai phong ~15-20GB dia
shutil.rmtree(work_dir, ignore_errors=True)
print("Da xoa thu muc repo tam (_src/) de giai phong dia - du lieu that nam trong train.bin/val.bin.")

# Bao dien thoai biet da xong - dung ntfy.sh (mien phi, khong can tai khoan)
try:
    import urllib.request as _ur
    _ur.urlopen(_ur.Request(
        "https://ntfy.sh/huyenb2404-vps-databuild-x7q2",
        data=f"XONG Buoc 1: {total_tokens:,} token da san sang trong train.bin/val.bin".encode(),
        method="POST",
    ), timeout=10)
except Exception as e:
    print(f"Khong gui duoc thong bao (khong sao, du lieu van xong binh thuong): {e}")
EOF
python3 data/code_3b/prepare.py
