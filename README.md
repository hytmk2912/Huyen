# Nền tảng AI tự động chạy cục bộ

Nền tảng AI chạy cục bộ, mọi thứ điều khiển bằng file cấu hình. Model chính đã xác minh là `huihui-ai/Huihui-Qwen3.8-27B-abliterated`; các model cục bộ khác được liệt kê trong `configs/models/platform.json` và được bộ định tuyến (router) chọn theo khả năng. Mặc định repo này không huấn luyện và không tải model 27B.

## Kiến trúc
- `local_ai.models`: cấu hình model, adapter, truy cập tokenizer và định tuyến theo khả năng.
- `local_ai.data`: một hệ thống dữ liệu/corpus thống nhất: ghi nguồn gốc dữ liệu, lọc, tokenize, chia shard, sổ đăng ký (registry) và lưu trữ Hugging Face tùy chọn.
- `local_ai.agents`: vòng lặp agent có giới hạn: lập kế hoạch → gọi công cụ → quan sát → đánh giá → thử lại.
- `local_ai.tools`: sổ đăng ký công cụ tường minh và giao diện sandbox cho phát triển.
- `local_ai.training`: kế hoạch huấn luyện và khung SFT tùy chọn (`finetune.py`, full hoặc LoRA).
- `local_ai.evaluation`, `local_ai.experiments`: các thành phần đánh giá và tái lập thí nghiệm, chạy theo cấu hình.

## Corpus
Mục tiêu là **10.000.000.000.000 token thật**. Đây chỉ là mục tiêu: token cục bộ hoặc token trong cấu hình không bao giờ được tính là tiến độ đã tải lên. Chỉ các shard production đã được xác minh trên máy chủ từ xa mới được tính. Bắt buộc có thông tin nguồn và giấy phép (license). Không có đợt thu thập dữ liệu lớn nào tự động chạy.

Tokenizer production phải được nạp từ model chính trong cấu hình. Các tokenizer thử nghiệm cũ (theo byte và `cl100k_base`) chỉ dùng cho test, không được tính vào corpus production.

## Model
`configs/models/platform.json` chứa danh sách model (`name`, `source` là tên trên Hugging Face, `dtype`, `capabilities`, và các thiết lập tùy chọn về revision/tokenizer/thiết bị). Mục đầu tiên là model chính. Muốn thêm model thì thêm một mục mới với `name` không trùng; router và khung huấn luyện đều gọi model theo tên.

## Huấn luyện (SFT)
Phần huấn luyện mới là khung tùy chọn: mặc định không huấn luyện gì, và test không tải model hay dataset nào.

1. Chuẩn bị dữ liệu từ một dataset trên Hugging Face. Sửa `configs/datasets/hf_sft.json`: điền `hf_dataset.name`, giấy phép lấy từ trang giới thiệu dataset (dataset card), và chọn một trong hai: `messages_field` (cột dạng hội thoại) hoặc `mapping.input` / `mapping.expected_output`. Sau đó chạy:

   ```bash
   python -m pip install datasets
   python -m local_ai.data hf-sft --config configs/datasets/hf_sft.json [--dataset org/name] [--limit 1000]
   ```

   Các dòng dữ liệu được chuyển sang schema của repo và đi qua các bước kiểm tra sẵn có: kiểm tra hợp lệ, loại trùng và chặn trùng với tập eval. Kết quả nằm trong `output_dir`: `raw.jsonl`, `train.jsonl`, `sft.jsonl`, `rejected.jsonl`, `manifest.json`. `HF_TOKEN` chỉ được đọc từ biến môi trường (cần khi dataset bị giới hạn truy cập hoặc riêng tư).

2. Fine-tune. `configs/training/sft.json` chỉ định model gốc (lấy theo tên trong danh sách model), đường dẫn `sft.jsonl`, thư mục đầu ra và các siêu tham số. `method` là `full` hoặc `lora`; gradient checkpointing bật sẵn.

   ```bash
   python -m pip install torch transformers trl peft datasets
   python -m local_ai.training.finetune --config configs/training/sft.json --dry-run
   python -m local_ai.training.finetune --config configs/training/sft.json --method lora
   ```

   Checkpoint được lưu thành `checkpoint-N` trong thư mục đầu ra; chạy lại sẽ tự tiếp tục từ checkpoint mới nhất (dùng `--no-resume` để bắt đầu lại). Adapter LoRA cuối cùng lưu vào `adapter/`, bản fine-tune full lưu vào `final/`. Nếu thiếu GPU hoặc thư viện tùy chọn, lượt chạy trả về `"status": "skipped"` thay vì báo lỗi.

## Cài đặt và kiểm tra
Cài các thư viện tùy chọn khi cần nạp model:

```bash
python -m pip install transformers torch huggingface_hub
# thư viện thêm cho huấn luyện (tùy chọn): trl peft datasets
python -m unittest discover -s tests -v
python -m compileall -q local_ai
```

Chỉ đặt `HF_TOKEN` trong biến môi trường (dùng khi tải lên hoặc tải dataset bị giới hạn truy cập); `HF_DATASET_REPO` có thể đặt ở biến môi trường hoặc trong cấu hình không chứa bí mật. Tuyệt đối không commit khóa/mật khẩu. Chạy `python -m local_ai.data secret-scan` trước khi thu thập dữ liệu.

## Lộ trình
1. Cố định phiên bản và chạy thử model/tokenizer chính trên phần cứng đích.
2. Hoàn tất xác minh từ xa có xác thực cho một shard.
3. Thêm dần các adapter nguồn corpus đã được duyệt và có giấy phép.
4. Triển khai huấn luyện phân tán BF16; FP8 vẫn chưa hỗ trợ cho đến khi đã chọn và thử nghiệm runtime cùng phần cứng.
5. Mở rộng công cụ cho agent, cơ chế cách ly, bộ đánh giá và kiểm thử khả năng phục hồi.
