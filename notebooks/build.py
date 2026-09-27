"""Sinh các notebook Colab trong thư mục này (lưu không kèm output). Sửa nội dung ô ở đây rồi chạy: python notebooks/build.py

Chỉ dùng thư viện chuẩn; test `tests/test_m10_colab.py` kiểm tra file .ipynb luôn khớp với nội dung sinh ra từ đây.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/hytmk2912/Huyen.git"
# Phiên bản đã chạy thử trong repo (xem README). Không cài lại torch: Colab có sẵn bản hợp với GPU.
PINNED = "transformers==5.17.0 trl==1.13.0 peft==0.21.0 datasets==5.0.1 accelerate==1.15.0 bitsandbytes==0.50.2"
# Notebook agent: Ollama bản mới nhất lúc viết (25/9/2026) và model qwen3:4b (khoảng 2,5 GB), trùng mục ollama-colab trong configs/models/platform.json.
OLLAMA_VERSION = "0.34.4"
OLLAMA_MODEL = "qwen3:4b"
# Notebook agent_trained_colab (M18): llama.cpp bản ghim chỉ dùng convert_hf_to_gguf.py (đã thử trên máy phát triển ngày 27/9 với
# model tí hon Qwen2, Qwen3 dùng tokenizer thật); thư viện gộp adapter cùng bản với notebook train, thêm sentencepiece cho llama.cpp.
LLAMA_CPP_TAG = "b11205"
EXPORT_PINNED = "transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 sentencepiece==0.2.2"


def markdown(cell_id: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(cell_id: str, text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict], name: str) -> dict:
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "name": name, "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def train_colab() -> dict:
    return notebook([
        markdown("gioi-thieu", """
# Train model nhỏ trên Colab miễn phí (GPU T4)

Notebook này fine-tune một model bằng QLoRA trên GPU T4 miễn phí của Colab, rồi so sánh điểm trước và sau khi train. Có 2 lựa chọn (đổi ở Bước 1):
- **smoke** (Qwen2.5-0.5B): nhanh, khoảng 30 phút cả notebook. Nên chạy thử cái này trước.
- **light** (Qwen3-4B): khoảng 2 giờ. Colab miễn phí hay ngắt giữa chừng; khi đó chỉ cần mở lại và Run all, việc train tự chạy tiếp.

Thời gian trên chỉ là ước lượng, chưa đo trên T4 thật; Bước 6 in ước tính chi tiết.

**Cần chuẩn bị (làm một lần):**
1. Tài khoản [Hugging Face](https://huggingface.co) và một token quyền **Write** (Settings → Access Tokens).
2. Trong Colab, bấm biểu tượng chìa khóa 🔑 (Secrets) ở cột bên trái → thêm secret tên `HF_TOKEN`, dán token vào, bật **Notebook access**.

**Cách chạy:** chọn model ở Bước 1, rồi vào menu Runtime → Run all (Chạy tất cả).

Hướng dẫn từng bước trên iPhone: [docs/TRAIN_COLAB.md](https://github.com/hytmk2912/Huyen/blob/main/docs/TRAIN_COLAB.md).
"""),
        code("buoc-1-chon-model", """
# Bước 1: chọn model rồi mới bấm Run all. Chạm vào ô chọn ở dòng MODEL (bên phải) để đổi.
# smoke = Qwen2.5-0.5B: nhanh, nên chạy thử trước. light = Qwen3-4B: lâu hơn nhiều, Colab ngắt thì Run all lại để train tiếp.
MODEL = "smoke"  # @param ["smoke", "light"]

# Độ dài tối đa mỗi câu trả lời khi chấm. Qwen3 (light) cần thêm chỗ cho phần suy nghĩ <think> trước câu trả lời.
MAX_NEW_TOKENS = {"smoke": 256, "light": 512}[MODEL]
print("Đã chọn model:", MODEL, "| cấu hình train: configs/training/colab_" + MODEL + ".json")
"""),
        code("buoc-2-gpu", """
# Bước 2: kiểm tra GPU. Colab miễn phí có GPU T4 (khoảng 15 GB).
# Nếu báo "Chưa có GPU": menu Runtime → Change runtime type → chọn "T4 GPU" → Save, rồi Run all lại.
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Chưa có GPU. Vào menu Runtime → Change runtime type → chọn T4 GPU → Save, rồi bấm Run all lại.")
print("GPU:", torch.cuda.get_device_name(0), "| bộ nhớ:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
"""),
        code("buoc-3-cai-dat", f"""
# Bước 3: tải code của repo và cài thư viện (đã ghim phiên bản; không cài lại torch).
# Chạy lại notebook thì chỉ cập nhật code mới nhất, không tải lại từ đầu. Từ đây, lệnh nào lỗi thì Run all dừng ở ô đó.
import os

if not os.path.isdir("/content/Huyen/local_ai"):
    !git clone --depth 1 {REPO_URL} /content/Huyen
%cd /content/Huyen
if not os.path.isdir("local_ai"):
    raise RuntimeError("Tải code thất bại: chưa có thư mục /content/Huyen/local_ai. Kiểm tra mạng rồi chạy lại ô này.")
from local_ai.colab import run  # chạy lệnh: lệnh lỗi thì ô báo đỏ và Run all dừng ở đúng chỗ lỗi

run("git pull --ff-only", "Bước 3 (cập nhật code)")
run("pip install -q {PINNED}", "Bước 3 (cài thư viện)")
# Colab cài sẵn torchao; bản đó không hợp với các thư viện đã ghim ở trên và làm lỗi khi train (chủ repo gặp khi chạy
# smoke và light ngày 25/9, phải gỡ tay). Repo không dùng torchao nên gỡ đi; máy chưa có torchao thì lệnh này chỉ báo bỏ qua.
run("pip uninstall -y -q torchao", "Bước 3 (gỡ torchao)")
"""),
        code("buoc-4-token", """
# Bước 4: lấy HF_TOKEN từ Colab Secrets (biểu tượng chìa khóa 🔑). Token không bị in ra và không lưu vào notebook.
import os
from google.colab import userdata

try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception as error:
    raise RuntimeError("Chưa đọc được HF_TOKEN: hãy thêm secret tên HF_TOKEN và bật Notebook access (" + type(error).__name__ + ")") from None

from huggingface_hub import whoami

HF_USER = whoami()["name"]
HUB_REPO = f"{HF_USER}/huyen-{MODEL}-qlora"  # repo riêng tư nhận checkpoint và adapter; muốn đổi tên thì sửa ở đây
print("Tài khoản Hugging Face:", HF_USER, "| repo:", HUB_REPO)
"""),
        code("buoc-5-du-lieu", """
# Bước 5: lấy 2000 dòng dữ liệu: 1800 dòng từ 3 preset (code 40%, lập luận 30%, tiếng Việt 30%) và 200 dòng (10%) gọi công cụ tự sinh
# (--tool-calls 0.1: trả JSON {"tool": ..., "arguments": ...} như câu tool_use của bộ chấm, có cả câu không cần công cụ), để model không quên cách gọi công cụ.
# Đọc kiểu streaming nên không tải cả dataset. Kết quả nằm ở data/processed/hf_sft/sft.jsonl.
# Bộ lọc chất lượng (ngôn ngữ, độ dài, lặp, gần trùng) và bộ chặn gần trùng với bộ chấm bỏ các dòng không đạt, nên có thể còn ít hơn 2000 dòng; số dòng bị bỏ ghi trong manifest.json.
run("python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --total 2000 --tool-calls 0.1 --output data/processed/hf_sft", "Bước 5 (lấy dữ liệu)")
"""),
        code("buoc-6-uoc-tinh", """
# Bước 6: ước tính thời gian train và chấm trên T4 (ước lượng thô, chưa đo trên T4 thật).
# Nếu Colab từng ngắt giữa chừng, ô này cho biết đã train được bao nhiêu bước; Bước 8 sẽ tự train tiếp từ đó.
run(f"python -m local_ai.training.estimate --config configs/training/colab_{MODEL}.json --max-new-tokens {MAX_NEW_TOKENS} --hub-model-id {HUB_REPO}", "Bước 6 (ước tính)")
"""),
        code("buoc-7-cham-truoc", """
# Bước 7: chấm model gốc (chưa train) trên bộ 38 câu eval, để lát nữa so sánh.
# --train-data kiểm tra dữ liệu train không chứa câu hỏi của bộ eval. --no-thinking tắt phần suy nghĩ <think> của Qwen3
# (Bước 9 dùng đúng cài đặt này). Chấm luôn greedy nên chạy lại ra cùng kết quả. Chấm xong, báo cáo được lưu vào repo riêng tư:
# Colab ngắt rồi chạy lại thì dùng lại báo cáo đó (cùng model, cùng cài đặt), không phải chấm lại.
run(f"python -m local_ai.evaluation --model {MODEL} --max-new-tokens {MAX_NEW_TOKENS} --no-thinking --train-data data/processed/hf_sft/sft.jsonl --output .runs/eval/{MODEL}/truoc --hub-repo {HUB_REPO} --hub-path eval/truoc", "Bước 7 (chấm trước)")
"""),
        code("buoc-8-train", """
# Bước 8: train QLoRA (nén 4bit, fp16 vì T4 không có bf16). Checkpoint được đẩy lên repo riêng tư HUB_REPO sau mỗi vài chục bước.
# Colab ngắt giữa chừng? Mở lại notebook, giữ nguyên model đã chọn và Run all: lệnh tự tải last-checkpoint từ Hugging Face về rồi train tiếp.
# Báo hết bộ nhớ (CUDA out of memory)? Chọn smoke, hoặc giảm max_length / per_device_batch_size trong configs/training/colab_<model>.json.
run(f"python -m local_ai.training.finetune --config configs/training/colab_{MODEL}.json --push-to-hub --hub-model-id {HUB_REPO}", "Bước 8 (train)")
"""),
        code("buoc-9-cham-sau", """
# Bước 9: chấm lại model sau khi train (model gốc + adapter vừa train: mục smoke-colab hoặc light-colab trong configs/models/platform.json).
# Cùng cài đặt với Bước 7 (--no-thinking). Luôn chấm lại (--no-reuse) vì adapter có thể đã đổi; báo cáo được lưu vào repo riêng tư ở eval/sau.
run(f"python -m local_ai.evaluation --model {MODEL}-colab --max-new-tokens {MAX_NEW_TOKENS} --no-thinking --train-data data/processed/hf_sft/sft.jsonl --output .runs/eval/{MODEL}/sau --hub-repo {HUB_REPO} --hub-path eval/sau --no-reuse", "Bước 9 (chấm sau)")
"""),
        code("buoc-10-so-sanh", """
# Bước 10: in bảng so sánh điểm trước và sau khi train (theo nhóm câu và theo ngôn ngữ).
run(f"python -m local_ai.evaluation.compare .runs/eval/{MODEL}/truoc/report.json .runs/eval/{MODEL}/sau/report.json", "Bước 10 (so sánh)")
"""),
        code("buoc-11-day-adapter", """
# Bước 11: đẩy adapter lên repo riêng tư trên Hugging Face để dùng lại sau.
run(f"python -m local_ai.training.hub push-adapter --repo {HUB_REPO} --adapter-dir .runs/colab_{MODEL}/adapter", "Bước 11 (đẩy adapter)")
print("Xong! Adapter nằm ở https://huggingface.co/" + HUB_REPO + " (chỉ tài khoản của bạn xem được).")
"""),
        code("buoc-12-so-do", """
# Bước 12: số đo thật (thời gian train, VRAM, tốc độ chấm) so với ước tính ở Bước 6. Hãy chụp màn hình bảng này gửi lại để sửa ước tính.
# Số đo cũng được lưu vào repo riêng tư HUB_REPO (thư mục so_do/), nên mất Colab cũng không mất số đo.
run(f"python -m local_ai.training.calibrate --config configs/training/colab_{MODEL}.json --max-new-tokens {MAX_NEW_TOKENS} --eval-before .runs/eval/{MODEL}/truoc/report.json --eval-after .runs/eval/{MODEL}/sau/report.json --push-to-hub --hub-model-id {HUB_REPO}", "Bước 12 (số đo)")
"""),
        markdown("ket-qua", """
## Kết quả nằm ở đâu
- **Adapter và checkpoint:** repo riêng tư `https://huggingface.co/<tên-bạn>/huyen-<model>-qlora` (ví dụ `huyen-smoke-qlora`). Thư mục `last-checkpoint` dùng để train tiếp.
- **Điểm eval:** bảng ở Bước 10. File chi tiết nằm ở `.runs/eval/<model>/truoc/` và `.runs/eval/<model>/sau/` (mất khi Colab tắt, nên hãy chụp màn hình bảng so sánh).
- **Số đo thật:** bảng ở Bước 12 (thời gian, VRAM, tốc độ chấm so với ước tính), lưu thêm ở `so_do/<model>.json` trong repo riêng tư. Hãy chụp màn hình bảng này gửi lại.

## Lỗi hay gặp
- **"Chưa có GPU" hoặc không kết nối được GPU:** chọn T4 ở Runtime → Change runtime type. Hết lượt GPU miễn phí thì đợi vài giờ rồi thử lại.
- **Ô báo đỏ "Bước N lỗi (mã thoát ...)":** lệnh của bước đó lỗi nên Run all dừng lại, các ô sau chưa chạy. Đọc thông báo ngay phía trên dòng đỏ, sửa xong thì chạy lại từ ô đó.
- **"Chưa đọc được HF_TOKEN":** thêm secret `HF_TOKEN` và bật Notebook access.
- **401 / 403 khi đẩy lên Hugging Face:** token chưa có quyền Write.
- **CUDA out of memory:** chọn smoke, hoặc giảm `max_length` / `per_device_batch_size` trong `configs/training/colab_<model>.json`.
- **Colab ngắt khi đang train:** mở lại notebook, giữ nguyên model đã chọn, Run all. Bước 6 cho biết đã train được bao nhiêu bước; Bước 7 dùng lại báo cáo chấm trước đã lưu, không chấm lại.
"""),
    ], "train_colab.ipynb")


def agent_colab() -> dict:
    return notebook([
        markdown("gioi-thieu", f"""
# Agent chạy model thật trên Colab (Ollama + {OLLAMA_MODEL})

Notebook này cài Ollama, tải model `{OLLAMA_MODEL}` (khoảng 2,5 GB), rồi cho agent của repo làm 6 nhiệm vụ mẫu bằng 2 công cụ:
- **calculator**: tính biểu thức số học;
- **terminal** (TerminalTool): chạy lệnh trong danh sách cho phép như `ls`, `cat`, chỉ trong thư mục làm việc riêng của từng nhiệm vụ.

Agent gọi model qua adapter kiểu OpenAI của repo (`http://localhost:11434/v1`). Mỗi nhiệm vụ in trace (kế hoạch, công cụ đã gọi, kết quả), cuối cùng in tỉ lệ thành công.

**Không cần token Hugging Face.** Nên chọn GPU T4 (menu Runtime → Change runtime type); chạy trên CPU cũng được nhưng rất chậm.

**Cách chạy:** menu Runtime → Run all. Thời gian ước tính: cài đặt và tải model khoảng 5 phút, 6 nhiệm vụ khoảng 5–15 phút (qwen3 suy nghĩ trước khi trả lời). Con số này chưa đo trên Colab thật.
"""),
        code("buoc-1-gpu", """
# Bước 1: kiểm tra GPU. Ollama chạy được trên CPU nhưng chậm hơn nhiều.
# Nên chọn T4: menu Runtime → Change runtime type → T4 GPU → Save, rồi Run all lại.
import shutil
import subprocess

if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)
else:
    print("Chưa có GPU: vẫn chạy được trên CPU nhưng rất chậm. Nên chọn T4 GPU rồi Run all lại.")
"""),
        code("buoc-2-tai-code", f"""
# Bước 2: tải code của repo. Phần agent chỉ dùng thư viện có sẵn của Python, không cần cài thêm gói pip nào.
# Từ đây, lệnh nào lỗi thì ô báo đỏ và Run all dừng ở ô đó.
import os

if not os.path.isdir("/content/Huyen/local_ai"):
    !git clone --depth 1 {REPO_URL} /content/Huyen
%cd /content/Huyen
if not os.path.isdir("local_ai"):
    raise RuntimeError("Tải code thất bại: chưa có thư mục /content/Huyen/local_ai. Kiểm tra mạng rồi chạy lại ô này.")
from local_ai.colab import run  # chạy lệnh: lệnh lỗi thì ô báo đỏ và Run all dừng ở đúng chỗ lỗi

run("git pull --ff-only", "Bước 2 (cập nhật code)")
"""),
        code("buoc-3-cai-ollama", f"""
# Bước 3: cài Ollama bản đã ghim ({OLLAMA_VERSION}). zstd dùng để giải nén bản cài, pciutils giúp Ollama nhận ra GPU.
run("apt-get -qq install -y zstd pciutils", "Bước 3 (cài zstd)", quiet=True)
run("curl -fsSL -o /tmp/ollama_install.sh https://ollama.com/install.sh", "Bước 3 (tải bản cài Ollama)")
run(["sh", "/tmp/ollama_install.sh"], "Bước 3 (cài Ollama)", env={{"OLLAMA_VERSION": "{OLLAMA_VERSION}"}})
run("ollama --version", "Bước 3 (kiểm tra Ollama)")
"""),
        code("buoc-4-chay-ollama", """
# Bước 4: chạy máy chủ Ollama ở nền (log ghi vào /content/ollama.log) và đợi tới khi nó trả lời.
# Chạy lại notebook khi Ollama đang chạy thì không mở thêm máy chủ mới.
import subprocess
import time
import urllib.request


def ollama_ready():
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        return True
    except OSError:
        return False


if not ollama_ready():
    subprocess.Popen(["ollama", "serve"], stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    if ollama_ready():
        print("Ollama đã chạy.")
        break
    time.sleep(1)
else:
    raise RuntimeError("Ollama chưa chạy sau 60 giây; hãy xem file /content/ollama.log")
"""),
        code("buoc-5-tai-model", f"""
# Bước 5: tải model {OLLAMA_MODEL} (khoảng 2,5 GB). Chạy lại thì Ollama không tải lại.
run("ollama pull {OLLAMA_MODEL}", "Bước 5 (tải model)")
"""),
        code("buoc-6-kiem-tra", """
# Bước 6: kiểm tra cấu hình, chưa gọi model: 6 nhiệm vụ, công cụ mỗi nhiệm vụ cần, địa chỉ máy chủ Ollama.
run("python -m local_ai.agents.tasks --model ollama-colab --dry-run", "Bước 6 (kiểm tra cấu hình)")
"""),
        code("buoc-7-chay-agent", """
# Bước 7: cho agent làm 6 nhiệm vụ mẫu. Mỗi nhiệm vụ in trace (kế hoạch, công cụ đã gọi, kết quả); dòng cuối là tỉ lệ thành công.
# TerminalTool chỉ được bật trong thư mục làm việc riêng của từng nhiệm vụ (.runs/agent_tasks/<nhiệm vụ>/workspace).
run("python -m local_ai.agents.tasks --model ollama-colab --output .runs/agent_tasks/report.json", "Bước 7 (chạy agent)")
"""),
        markdown("ket-qua", """
## Kết quả nằm ở đâu
- **Trace và tỉ lệ thành công:** in ngay dưới Bước 7. Chi tiết (JSON) ở `.runs/agent_tasks/report.json`; file này mất khi Colab tắt, nên hãy chụp màn hình.
- Một nhiệm vụ chỉ tính là **đạt** khi câu trả lời đúng **và** agent đã thật sự gọi công cụ cần dùng (không tính trường hợp model tự đoán).
- Model nhỏ đôi khi trả JSON sai dạng hoặc quên gọi công cụ. Khi đó nhiệm vụ không đạt; đây là điều notebook muốn đo.

## Lỗi hay gặp
- **Ô báo đỏ "Bước N lỗi (mã thoát ...)":** Run all dừng ở ô đó; đọc thông báo ngay phía trên dòng đỏ, sửa xong thì chạy lại từ ô đó.
- **"Không kết nối được server model":** Ollama chưa chạy; chạy lại Bước 4 (xem `/content/ollama.log`).
- **"không trả lời trong 300 giây":** đang chạy trên CPU hoặc máy quá tải; chọn T4 GPU rồi Run all lại.
- **Bước 3 báo cần zstd:** chạy lại Bước 3 (lệnh apt-get cài zstd nằm ngay đầu ô).
- **`ollama pull` lỗi mạng:** chạy lại Bước 5; phần đã tải được giữ lại.
"""),
    ], "agent_colab.ipynb")


def agent_trained_colab() -> dict:
    return notebook([
        markdown("gioi-thieu", f"""
# Agent dùng model đã train (Ollama, so với model gốc)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/agent_trained_colab.ipynb)

Notebook này lấy adapter bạn đã train bằng `train_colab` (repo riêng tư `<tên>/huyen-<model>-qlora`), gộp vào model gốc, xuất file GGUF, tạo 2 model trong Ollama rồi cho agent của repo làm 6 nhiệm vụ mẫu **với từng model** và in bảng so sánh:
- **model gốc**: model gốc cùng revision, cùng cách xuất (GGUF `q8_0`), không có adapter;
- **model đã train**: model gốc + adapter đã gộp.

Hai model đi qua đúng cùng một đường (cùng revision, dtype, kiểu lượng tử, chat template), nên chênh lệch là do train, không do cách xuất.

**Cần chuẩn bị:** đã chạy `train_colab` tới Bước 11 (đẩy adapter) với đúng model sẽ chọn ở Bước 1; token Hugging Face trong Colab Secrets tên `HF_TOKEN` (bật Notebook access); GPU T4 (menu Runtime → Change runtime type).

**Cách chạy:** chọn model ở Bước 1 rồi Runtime → Run all. Thời gian (ước lượng, chưa đo trên Colab thật): `smoke` khoảng 15–20 phút; `light` khoảng 45–70 phút (tải model gốc 8 GB, xuất 2 file GGUF khoảng 4,3 GB mỗi file, rồi 2 lượt agent).
"""),
        code("buoc-1-chon-model", """
# Bước 1: chọn model đã train bằng notebook train_colab (smoke = Qwen2.5-0.5B, light = Qwen3-4B), rồi mới bấm Run all.
MODEL = "smoke"  # @param ["smoke", "light"]
print("Đã chọn model:", MODEL, "| adapter: .runs/colab_" + MODEL + "/adapter | model Ollama: huyen-" + MODEL + "-goc và huyen-" + MODEL + "-da-train")
"""),
        code("buoc-2-gpu", """
# Bước 2: kiểm tra GPU. Gộp adapter và chạy Ollama cần GPU T4 (khoảng 15 GB); trên CPU sẽ rất chậm.
# Nếu báo "Chưa có GPU": menu Runtime → Change runtime type → chọn "T4 GPU" → Save, rồi Run all lại.
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Chưa có GPU. Vào menu Runtime → Change runtime type → chọn T4 GPU → Save, rồi bấm Run all lại.")
print("GPU:", torch.cuda.get_device_name(0), "| bộ nhớ:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
"""),
        code("buoc-3-cai-dat", f"""
# Bước 3: tải code của repo, cài thư viện đã ghim (không cài lại torch), tải llama.cpp bản {LLAMA_CPP_TAG} (chỉ dùng công cụ chuyển sang GGUF)
# và cài Ollama bản {OLLAMA_VERSION}. Chạy lại notebook thì không tải lại những gì đã có. Từ đây, lệnh nào lỗi thì Run all dừng ở ô đó.
import os

if not os.path.isdir("/content/Huyen/local_ai"):
    !git clone --depth 1 {REPO_URL} /content/Huyen
%cd /content/Huyen
if not os.path.isdir("local_ai"):
    raise RuntimeError("Tải code thất bại: chưa có thư mục /content/Huyen/local_ai. Kiểm tra mạng rồi chạy lại ô này.")
from local_ai.colab import run  # chạy lệnh: lệnh lỗi thì ô báo đỏ và Run all dừng ở đúng chỗ lỗi

run("git pull --ff-only", "Bước 3 (cập nhật code)")
run("pip install -q {EXPORT_PINNED}", "Bước 3 (cài thư viện)")
# Colab cài sẵn torchao; bản đó không hợp với các thư viện đã ghim (gặp khi train ngày 25/9). Repo không dùng torchao nên gỡ đi.
run("pip uninstall -y -q torchao", "Bước 3 (gỡ torchao)")
if not os.path.isfile("/content/llama.cpp/convert_hf_to_gguf.py"):
    run(["git", "clone", "--depth", "1", "--branch", "{LLAMA_CPP_TAG}", "https://github.com/ggml-org/llama.cpp", "/content/llama.cpp"], "Bước 3 (tải llama.cpp)")
run("apt-get -qq install -y zstd pciutils", "Bước 3 (cài zstd)", quiet=True)
run("curl -fsSL -o /tmp/ollama_install.sh https://ollama.com/install.sh", "Bước 3 (tải bản cài Ollama)")
run(["sh", "/tmp/ollama_install.sh"], "Bước 3 (cài Ollama)", env={{"OLLAMA_VERSION": "{OLLAMA_VERSION}"}})
"""),
        code("buoc-4-token", """
# Bước 4: lấy HF_TOKEN từ Colab Secrets (biểu tượng chìa khóa 🔑) để tải adapter từ repo riêng tư. Token không bị in ra và không lưu vào notebook.
import os
from google.colab import userdata

try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception as error:
    raise RuntimeError("Chưa đọc được HF_TOKEN: hãy thêm secret tên HF_TOKEN và bật Notebook access (" + type(error).__name__ + ")") from None

from huggingface_hub import whoami

HF_USER = whoami()["name"]
HUB_REPO = f"{HF_USER}/huyen-{MODEL}-qlora"  # repo riêng tư mà train_colab đã đẩy adapter lên (Bước 11)
print("Tài khoản Hugging Face:", HF_USER, "| repo:", HUB_REPO)
"""),
        code("buoc-5-tai-adapter", """
# Bước 5: tải adapter đã train (chỉ các file adapter, vài chục MB) về đúng thư mục adapter_path của mục <model>-colab trong configs/models/platform.json.
run(f"python -m local_ai.training.hub pull-adapter --repo {HUB_REPO} --adapter-dir .runs/colab_{MODEL}/adapter", "Bước 5 (tải adapter)")
"""),
        code("buoc-6-xuat-gguf", """
# Bước 6: xuất 2 file GGUF (q8_0) cùng một cách: model gốc (--no-adapter) và model gốc + adapter đã gộp.
# Mỗi lần: nạp model gốc không nén theo revision đã ghim → (gộp adapter) → safetensors → GGUF bằng llama.cpp → Modelfile cho Ollama.
# Chạy lại notebook thì bỏ qua file đã xuất xong (có export_info.json).
import os

for name, extra in (("goc", "--no-adapter"), ("da-train", "")):
    if os.path.isfile(f".runs/gguf/{MODEL}-{name}/export_info.json"):
        print("Đã có", f".runs/gguf/{MODEL}-{name}/model.gguf", "- bỏ qua")
        continue
    run(f"python -m local_ai.training.export --model {MODEL}-colab {extra} --output .runs/gguf/{MODEL}-{name} --llama-cpp /content/llama.cpp", f"Bước 6 (xuất {name})")
"""),
        code("buoc-7-chay-ollama", """
# Bước 7: chạy máy chủ Ollama ở nền (log ghi vào /content/ollama.log) và đợi tới khi nó trả lời.
import subprocess
import time
import urllib.request


def ollama_ready():
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        return True
    except OSError:
        return False


if not ollama_ready():
    subprocess.Popen(["ollama", "serve"], stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    if ollama_ready():
        print("Ollama đã chạy.")
        break
    time.sleep(1)
else:
    raise RuntimeError("Ollama chưa chạy sau 60 giây; hãy xem file /content/ollama.log")
"""),
        code("buoc-8-tao-model", """
# Bước 8: tạo 2 model trong Ollama từ file GGUF và Modelfile vừa xuất (chat template và tham số giống model cùng họ trong thư viện Ollama).
run(f"ollama create huyen-{MODEL}-goc -f .runs/gguf/{MODEL}-goc/Modelfile", "Bước 8 (tạo model gốc)")
run(f"ollama create huyen-{MODEL}-da-train -f .runs/gguf/{MODEL}-da-train/Modelfile", "Bước 8 (tạo model đã train)")
run("ollama list", "Bước 8 (danh sách model)")
"""),
        code("buoc-9-agent-goc", """
# Bước 9: agent làm 6 nhiệm vụ mẫu với model gốc (mục ollama-<model>-goc). Mỗi nhiệm vụ in trace; dòng cuối là tỉ lệ thành công.
run(f"python -m local_ai.agents.tasks --model ollama-{MODEL}-goc --workspace .runs/agent_tasks/{MODEL}-goc --output .runs/agent_tasks/{MODEL}-goc.json", "Bước 9 (agent với model gốc)")
"""),
        code("buoc-10-agent-da-train", """
# Bước 10: agent làm đúng 6 nhiệm vụ đó với model đã train (mục ollama-<model>-da-train).
run(f"python -m local_ai.agents.tasks --model ollama-{MODEL}-da-train --workspace .runs/agent_tasks/{MODEL}-da-train --output .runs/agent_tasks/{MODEL}-da-train.json", "Bước 10 (agent với model đã train)")
"""),
        code("buoc-11-so-sanh", """
# Bước 11: bảng so sánh từng nhiệm vụ, tỉ lệ thành công, thời gian, nhiệm vụ mới đạt / mới trượt sau khi train.
run(f"python -m local_ai.agents.compare .runs/agent_tasks/{MODEL}-goc.json .runs/agent_tasks/{MODEL}-da-train.json --labels 'model gốc' 'model đã train' --output .runs/agent_tasks/{MODEL}-so-sanh.json", "Bước 11 (so sánh)")
"""),
        markdown("ket-qua", """
## Kết quả nằm ở đâu
- **Bảng so sánh:** in dưới Bước 11 (chi tiết JSON ở `.runs/agent_tasks/<model>-so-sanh.json`). Trace từng nhiệm vụ in dưới Bước 9 và 10. Các file này mất khi Colab tắt, nên hãy chụp màn hình Bước 11 gửi lại.
- Một nhiệm vụ chỉ tính là **đạt** khi câu trả lời đúng **và** agent đã thật sự gọi công cụ cần dùng.
- Agent gửi `temperature` 0 và `seed` cố định, nên chạy lại ra cùng kết quả; chênh lệch giữa 2 model là do train.

## Lỗi hay gặp
- **Bước 5 báo "chưa có adapter" hoặc "không thấy repo":** chưa chạy `train_colab` tới Bước 11 với model này, hoặc chọn nhầm model ở Bước 1.
- **Bước 6 báo thiếu sentencepiece:** chạy lại Bước 3.
- **Bước 6 hết bộ nhớ khi gộp `light`:** Runtime → Restart session rồi Run all (file đã xuất xong được giữ lại).
- **"Không kết nối được server model":** Ollama chưa chạy; chạy lại Bước 7 (xem `/content/ollama.log`).
- **Ô báo đỏ "Bước N lỗi (mã thoát ...)":** đọc thông báo ngay phía trên dòng đỏ, sửa xong thì chạy lại từ ô đó.
"""),
    ], "agent_trained_colab.ipynb")


NOTEBOOKS = {"train_colab.ipynb": train_colab, "agent_colab.ipynb": agent_colab, "agent_trained_colab.ipynb": agent_trained_colab}


def render(builder) -> str:
    return json.dumps(builder(), ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    for name, builder in NOTEBOOKS.items():
        (HERE / name).write_text(render(builder), encoding="utf-8")
        print("Đã ghi", HERE / name)


if __name__ == "__main__":
    main()
