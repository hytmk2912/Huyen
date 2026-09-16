# AI Code Model — Train Từ Số 0 (Local, Miễn Phí)

📋 Xem **[TIEP_THEO.md](TIEP_THEO.md)** để biết việc cần làm, ai làm gì, cần công cụ/tài liệu gì.

🗺️ Xem **[LO_TRINH.md](LO_TRINH.md)** để biết dự án đang ở giai đoạn nào, % hoàn thành, và thời gian ước tính từng giai đoạn.

📦 Sau khi Bước 1 xong mà còn dư đĩa/thời gian rảnh → chạy thêm **[step1b_more_data.sh](step1b_more_data.sh)** (lấp đầy đĩa còn trống bằng thêm dữ liệu, không xoá gì cũ).

**Nếu thiếu chỗ giữa checkpoint và dữ liệu, ưu tiên giữ CHECKPOINT** — dữ liệu tải lại free được (đúng script này), checkpoint (công GPU thật) thì không. Mỗi lần chạy Bước 1/1b, script tự ghi lại đã dùng nguồn nào vào file `data/code_3b/DATA_MANIFEST.txt` — xem file đó để biết chính xác cần tải lại gì nếu sau này xoá đi.

## Việc cần làm trước khi có thiết bị

**Bạn — chỉ 1 việc**: thuê VPS/máy tính (CPU cũng được để thử trước, có GPU thì train nhanh hơn). Có rồi thì quay lại chat, nhắn Claude 1 câu là đã có thiết bị.

**Sau khi có thiết bị, cách ít việc nhất cho bạn**: cài Claude Code ngay trên thiết bị đó (đăng nhập bằng tài khoản claude.ai hiện có — **không** khai báo `ANTHROPIC_API_KEY` để khỏi tốn phí API riêng), rồi chỉ cần bảo nó "làm theo README trong repo github.com/huyenb2404-ops/Huyen". Nó tự tải, tự cài, tự train, tự sửa nếu lỗi, không cần bạn copy dán từng lệnh.

**Cần công cụ/tài khoản gì**: VPS/GPU thuê (cần nhiều RAM CPU + ổ NVMe/SSD nhanh — xem mục "Phần cứng" bên dưới) + tài khoản claude.ai đang dùng + tài khoản GitHub đã có. Không cần đăng ký thêm dịch vụ nào khác.

## 2 bước (giống cách các AI như Claude/GPT được tạo ra)

- **Bước 1 — Pretrain (đang làm, file này)**: train model đoán token/code tiếp theo từ dữ liệu thật. Chưa biết "nghe lời" hay trả lời như trợ lý.
- **Bước 2 — Fine-tune/Alignment (chưa làm, để sau)**: dạy model biết nghe lời, trả lời hữu ích. Chỉ làm khi Bước 1 đã ra được model đoán code đủ tốt.

## Thực tế cần biết trước khi bắt đầu

- **"Từ số 0" kiểu không dùng dữ liệu gì cả** (tự học 100% qua self-discovery) đã thử và ước tính mất **hàng trăm năm** — không khả thi, đã bỏ qua hướng này.
- **Hướng đang làm**: train từ đầu (random init, ~3B tham số) trên dữ liệu thật quy mô lớn, dùng kỹ thuật DeepSpeed ZeRO-Infinity để chứa model to trong phần cứng đi thuê.
- **Rủi ro đã biết trước, chấp nhận đi tiếp**: 3B tham số lý tưởng cần ~60 tỷ token mới học tử tế; bản đầu ở đây nhắm ~3-5 tỷ token (thực tế tải/xử lý được trong 1 lần chạy) — vẫn thiếu so với lý tưởng, có thể chạy lại nhiều lần tăng dần lượng data sau. Phần DeepSpeed **chưa test được bằng máy thật** (AI không có GPU) — khả năng cần sửa lỗi khi chạy lần đầu là có thật, đã có bước chạy thử ngắn ở dưới để giảm rủi ro tốn tiền.

## Công cụ dùng (100% free/local)

- **nanoGPT** (github.com/karpathy/nanoGPT) — code định nghĩa model GPT, ngắn gọn dễ chỉnh.
- **DeepSpeed ZeRO-Infinity** — cho phép train model 3B trên 1 GPU thuê bằng cách đẩy phần optimizer/tham số chưa cần dùng ngay sang RAM CPU và ổ NVMe.
- **Dữ liệu** (quy mô lớn hơn hẳn bản thử nghiệm trước):
  - **Code Python thật**: clone ~30 repo mã nguồn mở nổi tiếng (Flask, Django, NumPy, Pandas, FastAPI, scikit-learn...).
  - **FineWeb-Edu** (free, Hugging Face, mẫu có sẵn 10 tỷ token) — văn bản chất lượng cao quy mô lớn, không chỉ vài ngàn bài như trước.
  - **Wikipedia tiếng Anh + tiếng Việt** (free, Hugging Face).
  - **TinyStories, GSM8K, MATH, ARC-Easy** (free, Hugging Face) — ngôn ngữ mạch lạc + suy luận toán (dễ đến khó) + khoa học.
  - **Commit sửa lỗi thật** (~12 repo, mở rộng từ 6 — lấy lại từ chính các repo đã clone) — ví dụ thật về code sai → sửa đúng, để model học cách "bắt lỗi và tự sửa" thay vì chỉ học code đúng sẵn.
  - **codeparrot/apps** (free, Hugging Face) — 10.000 bài toán lập trình thật có lời giải, nhiều mức độ khó — tư duy giải quyết vấn đề bằng code.
  - **Khái niệm trading/thị trường** (~40 bài Wikipedia chọn lọc đúng chủ đề: order, thanh khoản, phân tích kỹ thuật, crypto... + bộ hỏi-đáp tài chính `finance-alpaca`) — **chỉ khái niệm, chưa có dữ liệu nến/giá lịch sử** (việc đó để sau, khi cần).
  - Đã giảm tỷ trọng FineWeb-Edu (từ 80% xuống 55% ngân sách) để nhường chỗ cho các nguồn suy luận ở trên — tăng mật độ "tư duy" trong cùng ngân sách token, thay vì chỉ tăng tổng số token.
- Không gọi API AI nào, không tốn phí ngoài tiền thuê GPU/VPS.

## Phần cứng cần

- **VRAM GPU**: từ 16GB trở lên là chạy được (ZeRO-Infinity đẩy bớt gánh nặng sang CPU/NVMe).
- **RAM CPU**: càng nhiều càng an toàn. Kiểm tra bằng `free -h`, báo AI biết số cụ thể để tính lại chính xác nếu cần.
- **Ổ NVMe/SSD tốc độ cao**: quyết định tốc độ train trực tiếp — ổ mạng/chia sẻ sẽ train chậm hơn nhiều, không phải lỗi cấu hình.
- **Dung lượng đĩa trống**: cần vài trăm GB cho dữ liệu — script bên dưới tự kiểm tra, báo lỗi rõ nếu thiếu thay vì tải nửa chừng rồi hỏng.

## Bước 1 — Chuẩn bị dữ liệu (~3-5 tỷ token)

```bash
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

# Ghi lai RO RANG da dung nguon nao, luc nao - de sau nay de kiem tra / tai lai dung y het
# neu can (vd sau khi xoa data de nhuong cho checkpoint).
import datetime
with open(os.path.join(os.path.dirname(__file__), "DATA_MANIFEST.txt"), "a") as mf:
    mf.write(f"[Buoc 1 - {datetime.datetime.now().isoformat()}]\n")
    mf.write(f"Tong token: {total_tokens:,}\n")
    mf.write("Nguon da dung (chi tiet xem prepare.py trong repo):\n")
    mf.write(f"  - Code: {len(REPOS)} repo GitHub (Flask, Django, NumPy, Pandas, ...)\n")
    mf.write("  - HuggingFaceFW/fineweb-edu (sample-10BT)\n")
    mf.write("  - wikimedia/wikipedia (20231101.en + 20231101.vi)\n")
    mf.write("  - roneneldan/TinyStories\n")
    mf.write("  - openai/gsm8k (main), hendrycks/competition_math, allenai/ai2_arc (ARC-Easy)\n")
    mf.write(f"  - Commit sua loi tu {len(HISTORY_REPOS)} repo (git history that)\n")
    mf.write("  - ~40 bai Wikipedia ve trading/thi truong + gbharti/finance-alpaca\n")
    mf.write("  - codeparrot/apps\n\n")
print("Da ghi DATA_MANIFEST.txt - de kiem tra sau nay dung nguon nao da dung.")

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
```

⚠️ Bước này có thể chạy **nhiều giờ** (tải + xử lý hàng tỷ token) — nên chạy trên VPS thường (CPU, rẻ hơn) trước, xong data mới thuê GPU để train, đỡ tốn tiền GPU trong lúc chờ tải.

## Bước 2 — Cài DeepSpeed + train model ~3B

**Chia nhiều lần thuê GPU — làm được, và nên làm vậy**: thuê GPU vài giờ, train, tắt máy, vài ngày sau thuê lại, script **tự tiếp tục từ checkpoint cũ**, không train lại từ đầu. Dữ liệu (`train.bin`/`val.bin` đã chuẩn bị ở Bước 1) dùng lại y nguyên mỗi lần — **không cần xoá đi tải lại**, vì mục tiêu là train nhiều lượt qua CÙNG 1 bộ dữ liệu đó (15 tỷ token) để model học sâu hơn dần, không phải mỗi lần đổi sang dữ liệu mới.

⚠️ **Kiểm tra 1 điều trước khi tắt máy GPU**: nếu nơi thuê GPU là máy **tạm thời, xoá hẳn sau khi trả** (không phải chỉ tắt/pause), checkpoint (`out-code-3b/`) sẽ mất theo — cần copy nó về VPS không GPU (dùng `scp -r out-code-3b/ user@vps-cu:~/ai-agent/nanoGPT/`) TRƯỚC khi trả máy GPU, rồi copy ngược lại khi thuê GPU lần sau. Nếu nơi thuê cho phép tạm dừng (giữ nguyên ổ đĩa, chỉ ngừng tính tiền GPU) thì không cần bước này. Không chắc loại nào thì cứ copy về cho chắc, chỉ mất thêm vài phút.

```bash
cd ~/ai-agent/nanoGPT

cat > ds_config.json << 'EOF'
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 32,
  "fp16": { "enabled": true },
  "optimizer": { "type": "AdamW", "params": { "lr": 3e-4, "betas": [0.9, 0.95] } },
  "scheduler": { "type": "WarmupLR", "params": { "warmup_min_lr": 0, "warmup_max_lr": 3e-4, "warmup_num_steps": 500 } },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "offload_param": { "device": "nvme", "nvme_path": "/root/ai-agent/nvme_offload", "pin_memory": true },
    "offload_optimizer": { "device": "nvme", "nvme_path": "/root/ai-agent/nvme_offload", "pin_memory": true },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e8,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5
  }
}
EOF
mkdir -p /root/ai-agent/nvme_offload

cat > train_deepspeed.py << 'EOF'
import os
import time
import numpy as np
import torch
import deepspeed
from model import GPTConfig, GPT

DATA_DIR = "data/code_3b"
BLOCK_SIZE = 1024        # tang tu 512 - xu ly duoc file code dai hon. Smoke test se cho biet
                         # co vua bo nho khong; neu OOM, giam MICRO_BATCH xuong 1 truoc (xem duoi)
MICRO_BATCH = 2          # phai khop voi train_micro_batch_size_per_gpu trong ds_config.json
GRAD_ACCUM = 32          # phai khop voi gradient_accumulation_steps trong ds_config.json
TARGET_HOURS = 3.0       # SUA SO NAY theo so gio ban dinh thue GPU - train se tu dung dung gio
LOG_EVERY = 10
SAVE_EVERY_SEC = 900     # luu checkpoint moi 15 phut, mat dien/dut ket noi khong mat het

def get_batch(split):
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (MICRO_BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x, y

train_size = os.path.getsize(os.path.join(DATA_DIR, "train.bin")) // 2  # uint16 = 2 byte/token
tokens_per_step = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
print(f"Du lieu train: {train_size:,} token. Moi step xu ly {tokens_per_step:,} token.")
print(f"De train het 1 luot du lieu can khoang {train_size / tokens_per_step:,.0f} step "
      f"(chua tinh toc do that cua may - se do o duoi).")

config = GPTConfig(block_size=BLOCK_SIZE, vocab_size=50304, n_layer=32,
                    n_head=20, n_embd=2560, dropout=0.1, bias=False)
model = GPT(config)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config="ds_config.json"
)

# QUAN TRONG: neu da tung train truoc do (co checkpoint trong out-code-3b), tiep tuc tu do -
# khong thi moi lan thue GPU lai la train lai tu dau, mat het cong suc lan truoc.
it = 0
tokens_seen = 0
try:
    load_path, client_state = model_engine.load_checkpoint("out-code-3b", tag="latest")
except Exception:
    load_path = None
if load_path is not None:
    it = client_state.get("it", 0)
    tokens_seen = client_state.get("tokens_seen", 0)
    print(f"Da tim thay checkpoint cu - TIEP TUC tu iter {it:,} ({tokens_seen:,} token da train truoc do).")
else:
    print("Khong co checkpoint cu - bat dau train tu dau (lan dau tien).")

start = time.time()
last_save = start
calibrated = False

while True:
    elapsed_hr = (time.time() - start) / 3600
    if elapsed_hr >= TARGET_HOURS:
        print(f"Da du {TARGET_HOURS} gio dat muc tieu, dung lai.")
        break

    x, y = get_batch("train")
    x, y = x.to(model_engine.device), y.to(model_engine.device)
    _, loss = model_engine(x, y)
    model_engine.backward(loss)
    model_engine.step()
    it += 1
    tokens_seen += tokens_per_step

    if it == 10 and not calibrated:
        sec_per_step = (time.time() - start) / 10
        total_steps_est = int(TARGET_HOURS * 3600 / sec_per_step)
        coverage = min(100, 100 * (total_steps_est * tokens_per_step) / train_size)
        print(f"\n[DO TOC DO] ~{sec_per_step:.1f} giay/step -> uoc tinh {total_steps_est:,} step "
              f"trong {TARGET_HOURS} gio -> se XU LY qua luong token tuong duong ~{coverage:.1f}% du lieu.\n"
              f"(Day la uoc tinh, khong phai % chinh xac: lay mau NGAU NHIEN nen co vung bi lap lai, "
              f"vung khac chua duoc thay - binh thuong voi cach train nay, khong phai loi.)\n")
        calibrated = True

    if it % LOG_EVERY == 0:
        print(f"iter {it} ({elapsed_hr:.2f}h): loss {loss.item():.4f}, {tokens_seen:,} token da qua")

    if time.time() - last_save >= SAVE_EVERY_SEC:
        model_engine.save_checkpoint("out-code-3b", tag="latest", client_state={"it": it, "tokens_seen": tokens_seen})
        last_save = time.time()

model_engine.save_checkpoint("out-code-3b", tag="latest", client_state={"it": it, "tokens_seen": tokens_seen})
print(f"Xong. Tong {it:,} step, {tokens_seen:,} token da train qua (cong don tu truoc den gio).")

try:
    import urllib.request as _ur
    _ur.urlopen(_ur.Request(
        "https://ntfy.sh/huyenb2404-vps-databuild-x7q2",
        data=f"XONG 1 phien train: {it:,} step, {tokens_seen:,} token cong don. Co the tat GPU.".encode(),
        method="POST",
    ), timeout=10)
except Exception as e:
    print(f"Khong gui duoc thong bao (khong sao): {e}")
EOF

```

### BẮT BUỘC chạy thử ngắn trước khi chạy thật

Sửa tạm `TARGET_HOURS = 3.0` thành `TARGET_HOURS = 0.03` (~2 phút) trong `train_deepspeed.py`, chạy:
```bash
deepspeed train_deepspeed.py
```
Không lỗi + thấy dòng `[DO TOC DO]` và `loss` in ra → sửa lại `TARGET_HOURS` thành số giờ thật bạn định thuê GPU rồi chạy thật (script tự dừng đúng giờ, tự tính đang train qua bao nhiêu % dữ liệu đã chuẩn bị — không cần đoán số iteration). Lỗi ngay ở bước này thì copy nguyên lỗi gửi AI ở đoạn chat mới — đỡ tốn tiền GPU cho 1 lỗi cấu hình.

Bước này cũng chính là chỗ kiểm tra `BLOCK_SIZE=1024` có vừa bộ nhớ máy bạn không (mới tăng từ 512). Nếu lỗi "out of memory" ở đây, sửa `MICRO_BATCH` từ 2 xuống 1 trong `train_deepspeed.py` rồi thử lại — vẫn chưa được thì hạ tiếp `BLOCK_SIZE` xuống 768.

## Sau khi train xong

DeepSpeed lưu checkpoint dạng riêng, cần đổi về dạng thường trước khi thử model:
```bash
cd ~/ai-agent/nanoGPT
python3 out-code-3b/zero_to_fp32.py out-code-3b model_fp32.pt

cat > sample_3b.py << 'EOF'
import torch
import tiktoken
from model import GPTConfig, GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("CANH BAO: khong thay GPU - sinh chu tren CPU voi model 3B se RAT cham (hang chuc giay/token). "
          "Chi nen dung de test nhanh vai token, khong dung de xai that.")

config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=32, n_head=20, n_embd=2560, dropout=0.0, bias=False)
model = GPT(config)
state_dict = torch.load("model_fp32.pt", map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
if device == "cuda":
    model = model.half()
model.eval()

enc = tiktoken.get_encoding("gpt2")

# Test qua du cac mang da train - khong chi code, de biet du lieu them vao co
# thuc su "vao" duoc khong (test 1 cau "def " thoi khong noi len duoc gi ve
# toan/trading/tieng Viet/tu duy sua loi).
PROMPTS = {
    "Code (Python)": "def fibonacci(n):",
    "Toan (GSM8K-style)": "Question: A store has 15 apples. It sells 6 and gets 20 more. How many apples now?\nAnswer:",
    "Tieng Viet": "Thanh pho Ha Noi la",
    "Trading/thi truong": "A limit order is",
    "Bat loi code": "# Bug: this function crashes on empty list\ndef get_first(items):\n    return items[0]\n\n# Fixed version:\ndef get_first(items):",
}

for label, prompt in PROMPTS.items():
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=50)
    print(f"\n=== {label} ===")
    print(enc.decode(out[0].tolist()))

print("\nLuu y: day chi la xem model sinh chu co hop ly khong bang mat, chua phai benchmark "
      "chinh xac (nhu HumanEval cho code hay dap so dung/sai cho toan) - lam benchmark that "
      "la buoc rieng, phuc tap hon, co the lam sau khi model on dinh hon.")
EOF
python3 sample_3b.py
```

## Đường đi tiếp theo

1. Model đầu tiên vẫn sẽ thiếu data so với mức lý tưởng (~60 tỷ token) — muốn tốt hơn thì tăng `TARGET_TOKENS` trong `prepare.py` và chạy lại (cần thêm đĩa + thời gian tải tương ứng).
2. Khi model đủ tốt để sinh code hợp lệ, mới nên nối nó với 1 vòng lặp tự động (giao việc → sinh code → test → sửa).
3. Train từ đầu là quá trình lặp đi lặp lại (thử → xem kết quả → chỉnh → thử lại), không phải 1 lần chạy là xong.
4. **Ý cho sau này — tách model trading riêng**: đã cân nhắc và tạm KHÔNG làm ngay (xem lý do bên dưới). Chỉ nên làm khi đủ 2 điều kiện:
   - Đã gom được nhiều dữ liệu trading hơn hẳn hiện tại (hiện chỉ ~40 bài Wikipedia + finance-alpaca, ~15-20 triệu token) — bao gồm cả dữ liệu nến/giá lịch sử (cố tình để sau, xem phần "Thực tế cần biết").
   - Model code chính (~2,7B) đã chạy thử thành công, có kết quả thật để đối chiếu.

   Lý do không tách ngay: chia 1 model 30B thành "15B code + 15B trading" KHÔNG giảm được vấn đề bộ nhớ (mỗi model 15B vẫn cần bộ nhớ train riêng của nó, không phải 1 nửa của 30B) và làm vấn đề dữ liệu tệ hơn (cần dữ liệu đủ cho CẢ HAI model riêng biệt, ~300 tỷ token/model, thay vì chia sẻ 1 bộ). Khi đủ dữ liệu trading thật để tách, quy mô model trading nên tính theo đúng lượng dữ liệu lúc đó có, không chọn số tham số trước rồi tìm dữ liệu sau.
