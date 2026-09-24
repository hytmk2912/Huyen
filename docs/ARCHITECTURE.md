# Kiến trúc

## Phạm vi
Repo fine-tune model Qwen3 (Huihui) có sẵn bằng LoRA/QLoRA. Không huấn luyện từ đầu (pretrain), không chứa agent. Trọng số model, dataset và khóa truy cập nằm ngoài repo, được chọn qua file cấu hình.

## Luồng xử lý

```
Hugging Face dataset
   │  local_ai/data/hub.py        tải (HF_TOKEN từ biến môi trường), chuyển sang schema
   ▼
bản ghi DatasetExample            local_ai/data/schema.py
   │  local_ai/data/core.py       kiểm tra → loại trùng → chặn rò rỉ eval → sft.jsonl
   ▼
sft.jsonl (mảng messages)
   │  local_ai/training/finetune.py   full / LoRA / QLoRA 4-bit, checkpoint, chạy tiếp
   ▼
adapter/ hoặc final/
   │  local_ai/evaluation/suites.py   chấm theo nhóm → eval_report.json cạnh checkpoint
   ▼
báo cáo đánh giá
```

## Các module
- `local_ai/data/core.py`: đọc JSON/JSONL/CSV/Parquet, kiểm tra schema, loại trùng theo `content_hash`, chặn rò rỉ eval, tạo `sft.jsonl`. Hàm `sft_messages` giữ nguyên hội thoại nhiều lượt; bản ghi một lượt thì context được đặt trước câu hỏi và reasoning nằm trong khối `<think>`.
- `local_ai/data/hub.py`: bước dữ liệu từ Hugging Face.
- `local_ai/data/secrets.py`: quét khóa/mật khẩu bị lộ (`python -m local_ai.data secret-scan`).
- `local_ai/models/adapters.py`: `ModelConfig`, `quantization_settings` (NF4 4-bit), adapter chạy suy luận.
- `local_ai/config/settings.py`: đọc danh sách model.
- `local_ai/training/finetune.py`: huấn luyện SFT bằng transformers + trl + peft (+ bitsandbytes cho QLoRA).
- `local_ai/evaluation/suites.py`: bộ đánh giá theo nhóm (khớp đúng, chứa đáp án, so số).
- `local_ai/experiments/tracking.py`: ghi cấu hình, metrics, checkpoint của lượt chạy.

## Nguyên tắc
- Thư viện nặng (torch, transformers, trl, peft, bitsandbytes) chỉ được import khi thật sự train. Thiếu GPU hoặc thư viện thì trả về `"skipped"`.
- Mọi test chạy được mà không cần mạng hay GPU.
