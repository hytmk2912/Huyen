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
# Chạy lại notebook thì chỉ cập nhật code mới nhất, không tải lại từ đầu.
import os

if not os.path.isdir("/content/Huyen"):
    !git clone --depth 1 {REPO_URL} /content/Huyen
else:
    !git -C /content/Huyen pull --ff-only
%cd /content/Huyen
!pip install -q {PINNED}
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
# Bước 5: lấy 2000 dòng dữ liệu từ 3 preset (code 40%, lập luận 30%, tiếng Việt 30%).
# Đọc kiểu streaming nên không tải cả dataset. Kết quả nằm ở data/processed/hf_sft/sft.jsonl.
!python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --total 2000 --output data/processed/hf_sft
"""),
        code("buoc-6-uoc-tinh", """
# Bước 6: ước tính thời gian train và chấm trên T4 (ước lượng thô, chưa đo trên T4 thật).
# Nếu Colab từng ngắt giữa chừng, ô này cho biết đã train được bao nhiêu bước; Bước 8 sẽ tự train tiếp từ đó.
!python -m local_ai.training.estimate --config configs/training/colab_{MODEL}.json --max-new-tokens {MAX_NEW_TOKENS} --hub-model-id {HUB_REPO}
"""),
        code("buoc-7-cham-truoc", """
# Bước 7: chấm model gốc (chưa train) trên bộ 30 câu eval, để lát nữa so sánh.
# --train-data kiểm tra dữ liệu train không chứa câu hỏi của bộ eval.
!python -m local_ai.evaluation --model {MODEL} --max-new-tokens {MAX_NEW_TOKENS} --train-data data/processed/hf_sft/sft.jsonl --output .runs/eval/{MODEL}/truoc
"""),
        code("buoc-8-train", """
# Bước 8: train QLoRA (nén 4bit, fp16 vì T4 không có bf16). Checkpoint được đẩy lên repo riêng tư HUB_REPO sau mỗi vài chục bước.
# Colab ngắt giữa chừng? Mở lại notebook, giữ nguyên model đã chọn và Run all: lệnh tự tải last-checkpoint từ Hugging Face về rồi train tiếp.
# Báo hết bộ nhớ (CUDA out of memory)? Chọn smoke, hoặc giảm max_length / per_device_batch_size trong configs/training/colab_<model>.json.
!python -m local_ai.training.finetune --config configs/training/colab_{MODEL}.json --push-to-hub --hub-model-id {HUB_REPO}
"""),
        code("buoc-9-cham-sau", """
# Bước 9: chấm lại model sau khi train (model gốc + adapter vừa train: mục smoke-colab hoặc light-colab trong configs/models/platform.json).
!python -m local_ai.evaluation --model {MODEL}-colab --max-new-tokens {MAX_NEW_TOKENS} --train-data data/processed/hf_sft/sft.jsonl --output .runs/eval/{MODEL}/sau
"""),
        code("buoc-10-so-sanh", """
# Bước 10: in bảng so sánh điểm trước và sau khi train (theo nhóm câu và theo ngôn ngữ).
!python -m local_ai.evaluation.compare .runs/eval/{MODEL}/truoc/report.json .runs/eval/{MODEL}/sau/report.json
"""),
        code("buoc-11-day-adapter", """
# Bước 11: đẩy adapter lên repo riêng tư trên Hugging Face để dùng lại sau.
!python -m local_ai.training.hub push-adapter --repo {HUB_REPO} --adapter-dir .runs/colab_{MODEL}/adapter
print("Xong! Adapter nằm ở https://huggingface.co/" + HUB_REPO + " (chỉ tài khoản của bạn xem được).")
"""),
        markdown("ket-qua", """
## Kết quả nằm ở đâu
- **Adapter và checkpoint:** repo riêng tư `https://huggingface.co/<tên-bạn>/huyen-<model>-qlora` (ví dụ `huyen-smoke-qlora`). Thư mục `last-checkpoint` dùng để train tiếp.
- **Điểm eval:** bảng ở Bước 10. File chi tiết nằm ở `.runs/eval/<model>/truoc/` và `.runs/eval/<model>/sau/` (mất khi Colab tắt, nên hãy chụp màn hình bảng so sánh).

## Lỗi hay gặp
- **"Chưa có GPU" hoặc không kết nối được GPU:** chọn T4 ở Runtime → Change runtime type. Hết lượt GPU miễn phí thì đợi vài giờ rồi thử lại.
- **"Chưa đọc được HF_TOKEN":** thêm secret `HF_TOKEN` và bật Notebook access.
- **401 / 403 khi đẩy lên Hugging Face:** token chưa có quyền Write.
- **CUDA out of memory:** chọn smoke, hoặc giảm `max_length` / `per_device_batch_size` trong `configs/training/colab_<model>.json`.
- **Colab ngắt khi đang train:** mở lại notebook, giữ nguyên model đã chọn, Run all. Bước 6 cho biết đã train được bao nhiêu bước.
"""),
    ], "train_colab.ipynb")


NOTEBOOKS = {"train_colab.ipynb": train_colab}


def render(builder) -> str:
    return json.dumps(builder(), ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    for name, builder in NOTEBOOKS.items():
        (HERE / name).write_text(render(builder), encoding="utf-8")
        print("Đã ghi", HERE / name)


if __name__ == "__main__":
    main()
