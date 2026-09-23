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
Mục tiêu là **10.000.000.000.000 token thật**, chia theo tỷ lệ trong `domain_mixture` của `configs/datasets/corpus_10t.json`:

| Nhóm (`domain`) | Tỷ lệ | Token mục tiêu |
| --- | --- | --- |
| `code` | 10% | 1.000 tỷ |
| `trading` | 20% | 2.000 tỷ |
| `reasoning` (suy luận) | 10% | 1.000 tỷ |
| `vietnamese` (tiếng Việt) | 10% | 1.000 tỷ |
| `general` (dữ liệu tự do) | 50% | 5.000 tỷ |

Mỗi nguồn dữ liệu phải thuộc một nhóm có trong bảng; nguồn thuộc nhóm khác sẽ bị từ chối. `python -m local_ai.data mixture --config configs/datasets/corpus_10t.json` in số token mục tiêu từng nhóm, còn `progress` báo tiến độ từng nhóm so với mục tiêu (`by_domain_target`).

 Đây chỉ là mục tiêu: token cục bộ hoặc token trong cấu hình không bao giờ được tính là tiến độ đã tải lên. Chỉ các shard production đã được xác minh trên máy chủ từ xa mới được tính. Bắt buộc có thông tin nguồn và giấy phép (license). Không có đợt thu thập dữ liệu lớn nào tự động chạy.

Tokenizer production phải được nạp từ model chính trong cấu hình. Các tokenizer thử nghiệm cũ (theo byte và `cl100k_base`) chỉ dùng cho test, không được tính vào corpus production.

## Model
`configs/models/platform.json` chứa danh sách model (`name`, `source` là tên trên Hugging Face, `dtype`, `capabilities`, và các thiết lập tùy chọn về revision/tokenizer/thiết bị). Mục đầu tiên là model chính. Muốn thêm model thì thêm một mục mới với `name` không trùng; router và khung huấn luyện đều gọi model theo tên.

Ba model phụ là dòng Huihui Qwen3 bản FP32 (`dtype: float32`) kèm file GGUF f32:

| Tên | Nguồn trên Hugging Face | Dung lượng GGUF f32 (ước tính) |
| --- | --- | --- |
| `huihui-qwen3-4b` | `huihui-ai/Huihui-Qwen3-4B-abliterated-v2` | ~16 GB |
| `huihui-qwen3-8b` | `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` | ~33 GB |
| `huihui-qwen3-14b` | `huihui-ai/Huihui-Qwen3-14B-abliterated-v2` | ~59 GB |

Dòng Qwen3 không có cỡ 3B và 7B nên dùng cỡ gần nhất là 4B và 8B. Trên Hugging Face chưa có bản GGUF f32 của các model này (cao nhất là f16), nên file GGUF f32 được xuất từ trọng số gốc bằng llama.cpp:

```bash
git clone https://github.com/ggml-org/llama.cpp
python -m pip install huggingface_hub -r llama.cpp/requirements.txt
python -m local_ai.models.gguf --model huihui-qwen3-4b --llama-cpp llama.cpp --dry-run
python -m local_ai.models.gguf --model huihui-qwen3-4b --llama-cpp llama.cpp
```

File được lưu vào đường dẫn `gguf_file` trong cấu hình (thư mục `models/gguf/`, không commit lên git). Khi file đã có, adapter nạp model trực tiếp từ GGUF.

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
**Quy tắc:** lộ trình luôn có đúng **5 mục đang làm**. Làm xong mục nào thì chuyển mục đó xuống "Đã hoàn thành" và bổ sung ngay một mục mới chưa làm, để lộ trình luôn đủ 5 mục. Với mục cần phần cứng hoặc khóa truy cập thật, phần code và test được làm trong repo; phần phải chạy thật được ghi vào "Việc cần chạy trên máy thật".

### Đang làm
1. Giới hạn tài nguyên khi thu thập corpus: dừng an toàn khi vượt `max_local_storage_gb`, tôn trọng `max_bandwidth_mbps`, và chạy tiếp được sau khi dừng.
2. Ngân sách cho agent: giới hạn thời gian mỗi lần gọi công cụ và tổng thời gian/token của một nhiệm vụ, dừng gọn khi vượt.
3. Tự động đánh giá sau huấn luyện: chạy bộ đánh giá theo nhóm sau mỗi lần fine-tune và lưu báo cáo cạnh checkpoint.
4. Cố định revision (mã commit) cho các nguồn `hf_dataset` và đo số token thật của từng nguồn bằng tokenizer production thay cho số ước lượng.
5. Tìm thêm nguồn trading thuần (tin thị trường, dữ liệu giá có giấy phép rõ) và nguồn cho nhóm `reasoning` để bám đủ tỷ lệ 20% và 10%.

### Đã hoàn thành
- **Loại trùng gần đúng (near-dedup)**: `local_ai/data/dedup.py` dùng MinHash + LSH, chia văn bản thành từng đoạn, loại đoạn giống ≥ 80% với đoạn đã có ở bất kỳ nguồn nào (chỉ số lưu trong registry). Xây lại một nguồn không bị tự coi là trùng; nguồn trùng toàn bộ bị từ chối (`near_duplicate`). Cấu hình ở `near_dedup` trong `corpus_10t.json`.
- **Nguồn thật cho trading và tiếng Việt** (đã khai báo, chưa tải): tiếng Việt lấy từ `HuggingFaceFW/fineweb-2` phần `vie_Latn` (ODC-By 1.0); trading lấy từ `PleIAs/SEC` (báo cáo 10-K, CC0-1.0). Báo cáo 10-K là dữ liệu tài chính doanh nghiệp, chưa phải dữ liệu giao dịch thuần, nên đã thêm mục tìm nguồn trading khác. Số token trong cấu hình chỉ là ước lượng, và mỗi lần xây đang giới hạn `max_rows` (20.000 dòng tiếng Việt, 500 báo cáo SEC).
- **Công cụ, cách ly và phục hồi cho agent**: công cụ `read_file`/`list_files`/`write_file` chỉ hoạt động trong thư mục làm việc (chặn `../`, giới hạn kích thước, ghi file phải bật riêng). Sandbox Python không truyền biến môi trường của máy (không lộ `HF_TOKEN`) và giới hạn bộ nhớ/CPU. Agent và công cụ không bị dừng khi model hoặc công cụ gặp lỗi bất ngờ.
- **Bộ đánh giá tiếng Việt và trading**: `local_ai/evaluation/suites.py` chấm theo 3 cách (khớp đúng, chứa đáp án, so số có sai số; hiểu cả số kiểu Việt Nam như `1.000.000` hay `12,5`), báo cáo độ chính xác theo từng nhóm. Bộ câu hỏi mẫu `data/eval/vi_trading_eval.jsonl` (4 câu tiếng Việt, 4 câu trading). Chạy: `python -m local_ai.evaluation.suites --model <tên>`.
- **Adapter nguồn corpus đã duyệt**: nguồn mới có thể lấy từ dataset Hugging Face (`download_method: "hf_dataset"`, `url: "hf://datasets/org/name"`, tải dạng streaming, chọn cột văn bản bằng `options.text_field`). `allowed_licenses` trong `corpus_10t.json` là danh sách giấy phép đã duyệt; nguồn có giấy phép khác bị từ chối. Chưa khai báo nguồn thật nào cho trading/tiếng Việt (đã đưa thành mục mới).
- **Huấn luyện phân tán BF16** (phần code): cấu hình `distributed` với `ddp` hoặc `fsdp`, huấn luyện phân tán luôn dùng bfloat16, FP8 bị từ chối. Mẫu `configs/training/sft_fsdp.json` (full fine-tune 14B trên 8 GPU); `--print-launch` in lệnh `torchrun` để chạy. Thiếu GPU hoặc không đủ số GPU thì báo `skipped`.
- **Cố định phiên bản và chạy thử model/tokenizer** (phần code): `python -m local_ai.models.smoke pin --model <tên>` ghi mã commit Hugging Face vào `revision`/`tokenizer_revision`; `python -m local_ai.models.smoke run --model <tên>` nạp model, sinh thử một câu và ghi báo cáo vào `.runs/smoke/`. Thiếu GPU/thư viện thì báo `skipped`.
- **Xác minh một shard trên máy chủ có xác thực** (phần code): `python -m local_ai.data verify-shard --config <file>` tải lên một shard `VALIDATED`, kiểm tra checksum, xác minh trên Hugging Face rồi đánh dấu `COMPLETE`. Thiếu `HF_TOKEN`/`HF_DATASET_REPO` thì báo `skipped`.

### Việc cần chạy trên máy thật
- Xuất GGUF f32 cho 3 model Huihui Qwen3 (cần llama.cpp và khoảng 110 GB ổ đĩa cho cả ba file).
- Cố định phiên bản rồi chạy thử model chính trên GPU: `python -m local_ai.models.smoke pin --model primary`, sau đó `python -m local_ai.models.smoke run --model primary` (cần GPU đủ bộ nhớ cho model 27B BF16).
- Chạy thử huấn luyện phân tán: `python -m local_ai.training.finetune --config configs/training/sft_fsdp.json --print-launch`, rồi chạy lệnh `torchrun` in ra trên máy có 8 GPU (cần chuẩn bị `sft.jsonl` trước).
- Chạy bộ đánh giá tiếng Việt/trading trên GPU: `python -m local_ai.evaluation.suites --model primary --report .runs/eval/primary.json`.
- Tải và xây corpus từ các nguồn mới: `python -m pip install datasets`, rồi `python -m local_ai.data build-corpus --config configs/datasets/corpus_10t.json` (cần mạng; nguồn Hugging Face tải dạng streaming).
- Xác minh một shard thật: đặt `HF_TOKEN` và `HF_DATASET_REPO` trong biến môi trường, chạy `build-corpus` rồi `python -m local_ai.data verify-shard --config configs/datasets/corpus_10t.json`.
