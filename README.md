# Fine-tune model Qwen3 (Huihui) chạy cục bộ

Repo dùng để fine-tune model có sẵn trên Hugging Face, mọi bước đều điều khiển bằng file cấu hình:

```
dataset Hugging Face → kiểm tra, loại trùng, chặn rò rỉ eval → sft.jsonl → fine-tune LoRA/QLoRA → đánh giá
```

Model chính là `huihui-ai/Huihui-Qwen3.8-27B-abliterated`; các model khác nằm trong `configs/models/platform.json`. Repo không tự tải model: test chạy được mà không cần mạng hay GPU.

Kế hoạch làm việc hiện tại: xem `TASKS.md` (nhiệm vụ 1 tuần, 7 mốc) và `memory.md` (mốc đang làm, việc dở, lỗi gặp).

## Cấu trúc
- `local_ai/data`: đọc dữ liệu, kiểm tra schema, loại trùng, chặn rò rỉ eval, xuất `sft.jsonl`; bước tải dataset Hugging Face (`hub.py`); quét khóa bí mật (`secrets.py`).
- `local_ai/training`: khung fine-tune SFT (`finetune.py`, full hoặc LoRA).
- `local_ai/models`: cấu hình model, adapter chạy model Hugging Face, router chọn model theo khả năng.
- `local_ai/evaluation`: chấm điểm model.
- `local_ai/agents`, `local_ai/tools`: agent có giới hạn vòng lặp và các công cụ (dùng để thử model); công cụ lỗi không làm sập agent.
- `archive/`: phần đã cất, không còn dùng (corpus 10T token, hướng dẫn nanoGPT cũ).

## Cài đặt và kiểm tra
```bash
python -m pip install datasets                        # tải dataset Hugging Face
python -m pip install torch transformers trl peft     # huấn luyện (cần GPU)
python -m unittest discover -s tests -v
python -m compileall -q local_ai
python -m local_ai.data secret-scan
```

Chỉ đặt `HF_TOKEN` trong biến môi trường hoặc file `.env` (đã có trong `.gitignore`); tuyệt đối không commit khóa hay mật khẩu.

## Bước 1: chuẩn bị dữ liệu
Sửa `configs/datasets/hf_sft.json`: điền `hf_dataset.name`, giấy phép lấy từ trang dataset, và chọn `messages_field` (cột hội thoại) hoặc `mapping.input` / `mapping.expected_output`.

```bash
python -m local_ai.data hf-sft --config configs/datasets/hf_sft.json [--dataset org/name] [--limit 1000]
```

Kết quả nằm trong `output_dir`: `sft.jsonl` (dùng để train), `train.jsonl`, `rejected.jsonl`, `manifest.json`.

## Bước 2: fine-tune
`configs/training/sft.json` chỉ định model gốc, đường dẫn `sft.jsonl`, thư mục đầu ra và siêu tham số; `method` là `full` hoặc `lora`.

```bash
python -m local_ai.training.finetune --config configs/training/sft.json --dry-run
python -m local_ai.training.finetune --config configs/training/sft.json
```

Checkpoint lưu thành `checkpoint-N`; chạy lại sẽ tiếp tục từ checkpoint mới nhất (`--no-resume` để làm lại). Thiếu GPU hoặc thư viện thì lệnh báo `"status": "skipped"`.

## Lộ trình
Theo `TASKS.md`: M1 agent chịu lỗi và dọn repo → M2 model ảnh+chữ, nén 4-bit (QLoRA), ước tính VRAM → M3 adapter server local (Ollama, llama.cpp) → M4 preset dataset → M5 eval mở rộng → M6 test chạy thật trên CPU → M7 tổng kết. Tiến độ từng mốc ghi trong bảng ở `TASKS.md`.
