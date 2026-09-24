# Fine-tune model Qwen3 (Huihui) chạy cục bộ

Repo dùng để fine-tune model có sẵn trên Hugging Face, mọi bước đều điều khiển bằng file cấu hình:

```
dataset Hugging Face → kiểm tra, loại trùng, chặn rò rỉ eval → sft.jsonl → fine-tune LoRA/QLoRA → đánh giá
```

Model chính là `huihui-ai/Huihui-Qwen3.8-27B-abliterated`; các model khác nằm trong `configs/models/platform.json`. Repo không tự tải model: test chạy được mà không cần mạng hay GPU.

Kế hoạch làm việc hiện tại: xem `TASKS.md` (nhiệm vụ 1 tuần, 7 mốc) và `memory.md` (mốc đang làm, việc dở, lỗi gặp).

## Cấu trúc
- `local_ai/data`: đọc dữ liệu, kiểm tra schema, loại trùng, chặn rò rỉ eval, xuất `sft.jsonl`; bước tải dataset Hugging Face (`hub.py`); quét khóa bí mật (`secrets.py`).
- `local_ai/training`: khung fine-tune SFT (`finetune.py`: full, LoRA hoặc QLoRA).
- `local_ai/models`: cấu hình model, adapter chạy model Hugging Face (chữ hoặc ảnh + chữ, nén 4bit/8bit), adapter gọi server local kiểu OpenAI (`openai_compatible.py`), router chọn model theo khả năng, ước tính VRAM (`vram.py`).
- `local_ai/evaluation`: bộ eval (`suite.py`) với 4 cách chấm, lệnh `python -m local_ai.evaluation`; câu hỏi nằm trong `data/eval/eval_v1.jsonl`.
- `local_ai/agents`, `local_ai/tools`: agent có giới hạn vòng lặp và các công cụ (dùng để thử model); công cụ lỗi không làm sập agent.
- `archive/`: phần đã cất, không còn dùng (corpus 10T token, hướng dẫn nanoGPT cũ).

## Cài đặt và kiểm tra
```bash
python -m pip install datasets                        # tải dataset Hugging Face
python -m pip install torch transformers trl peft     # huấn luyện (cần GPU)
# máy không có GPU: cài torch bản CPU trước, rồi mới cài các thư viện kia
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install bitsandbytes                    # nén 4bit/8bit và QLoRA (cần GPU CUDA)
python -m unittest discover -s tests -v
python -m compileall -q local_ai
python -m local_ai.data secret-scan
```

Chỉ đặt `HF_TOKEN` trong biến môi trường hoặc file `.env` (đã có trong `.gitignore`); tuyệt đối không commit khóa hay mật khẩu.

## Danh sách model
Mỗi model trong `configs/models/platform.json` khai báo:
- `kind`: `text` (nạp bằng `AutoTokenizer` + `AutoModelForCausalLM`) hoặc `multimodal` (ảnh + chữ, nạp bằng `AutoProcessor` + `AutoModelForMultimodalLM`; bản transformers cũ không có lớp này thì dùng `AutoModelForImageTextToText`). Model chính là `multimodal` theo thẻ trên Hugging Face.
- `params_b`: số tỷ tham số, dùng để ước tính VRAM.
- `quantization` (tùy chọn): `4bit` (NF4) hoặc `8bit`, nén bằng bitsandbytes khi nạp model.

GPU không hỗ trợ bf16 (ví dụ T4) thì model tự chuyển sang fp16 và in cảnh báo. Chỉ model `multimodal` nhận được ảnh (`Message(..., images=("duong/dan/anh.png",))`); gửi ảnh tới model `text` sẽ báo lỗi.

Xem VRAM ước tính cho mọi model (không tải model, không cần mạng):
```bash
python -m local_ai.models.vram
```
Bảng in ra trọng số ở bf16/8bit/4bit và bộ nhớ khi train LoRA/QLoRA. Ví dụ model chính (27,78 tỷ tham số): trọng số bf16 khoảng 55,6 GB, 4bit khoảng 15,6 GB, train QLoRA khoảng 20,5 GB. Đây chỉ là ước lượng, chưa đo trên GPU thật.

## Chạy model qua server local (Ollama, llama.cpp, vLLM)
Model có `backend: "openai_compatible"` được gọi qua `POST {base_url}/chat/completions` theo chuẩn OpenAI, chỉ dùng thư viện chuẩn của Python. Khi đó:
- `source` là tên model trên server;
- `timeout_s` là số giây chờ tối đa;
- `api_key_env` (tùy chọn) là **tên** biến môi trường chứa khóa, dùng khi server yêu cầu khóa. Không ghi khóa vào file cấu hình.

`base_url` chỉ được là máy này hoặc mạng nội bộ (localhost, 192.168.x.x, 10.x.x.x...). Địa chỉ Internet bị từ chối, để không vô tình gọi API trả phí.

`configs/models/platform.json` có sẵn 2 mục mẫu:
- `ollama`: `http://localhost:11434/v1`, model `huihui_ai/Qwen3.8-abliterated:27b`;
- `llamacpp`: `http://localhost:8080/v1`.

Cần cài Ollama, llama.cpp hoặc vLLM trước.

```bash
# Ollama
ollama pull huihui_ai/Qwen3.8-abliterated:27b
ollama serve                                   # nếu Ollama chưa tự chạy nền
python -m local_ai.demo --model ollama

# llama.cpp (model nhỏ, chạy được trên CPU)
llama-server -hf Qwen/Qwen2.5-0.5B-Instruct-GGUF --port 8080
python -m local_ai.demo --model llamacpp

# vLLM (cần GPU)
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

Với vLLM, thêm một mục vào `configs/models/platform.json`:
```json
{"name": "vllm", "backend": "openai_compatible", "base_url": "http://localhost:8000/v1", "source": "Qwen/Qwen2.5-0.5B-Instruct", "kind": "text", "params_b": 0.49, "capabilities": ["chat", "reasoning"]}
```
Nếu chạy `vllm serve ... --api-key "$LOCAL_LLM_API_KEY"` thì thêm `"api_key_env": "LOCAL_LLM_API_KEY"` vào mục trên.

`python -m local_ai.demo` (không có `--model`) chạy agent với model giả, không cần server.

Khi server lỗi, agent dừng lại và báo lỗi bằng tiếng Việt, không bị sập. Các trường hợp lỗi: server chưa chạy, hết thời gian chờ, hoặc server trả mã lỗi HTTP. Model quá nhỏ có thể không trả JSON đúng định dạng; khi đó demo báo "không hoàn thành".

## Bước 1: chuẩn bị dữ liệu

### Dùng preset có sẵn
`configs/datasets/presets/` có 3 preset trỏ tới dataset công khai trên Hugging Face. Mỗi preset:
- khóa cố định một commit (`revision`);
- đọc kiểu streaming, tối đa 1000 dòng, nên không tải cả dataset.

| Preset | Dataset | Nội dung | Giấy phép ghi trên Hugging Face |
| --- | --- | --- | --- |
| `code` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | Bài lập trình Python kèm lời giải đã chạy thử (tiếng Anh) | ODC-By |
| `reasoning` | `open-r1/OpenR1-Math-220k` | Bài toán + lời giải (đưa vào khối `<think>`) + đáp án (tiếng Anh) | Apache-2.0 |
| `vietnamese` | `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` | Hội thoại tiếng Việt nhiều lượt, giữ nguyên mọi lượt | Apache-2.0 |

**Trước khi dùng, hãy đọc lại giấy phép trên dataset card** (trang giới thiệu dataset). Giấy phép trong preset được ghi kèm trạng thái "cần kiểm tra lại trên dataset card", vì dữ liệu có thể được dịch hoặc sinh từ model khác, và giấy phép gốc có thể chặt hơn.

```bash
python -m local_ai.data list-presets                                          # xem các preset
python -m local_ai.data hf-sft --config configs/datasets/presets/code.json    # một preset
python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --output data/processed/hf_mix
```

Trộn nhiều preset (lặp lại `--preset tên:tỉ_lệ`) thì kết quả chung vào một `sft.jsonl`:
- Tỉ lệ được chuẩn hóa về tổng 1.
- Nếu không ghi `--total`, lệnh lấy tổng số dòng lớn nhất sao cho không preset nào vượt `limit` của nó.
- `--limit 20` thì mỗi preset chỉ đọc tối đa 20 dòng, hợp để chạy thử.
- Tỉ lệ tính trên số dòng đọc vào. Dòng bị loại (trùng, sai dạng) làm tỉ lệ cuối lệch nhẹ; số dòng thật của từng nguồn ghi trong `manifest.json`.

### Tự khai báo dataset
Sửa `configs/datasets/hf_sft.json`: điền `hf_dataset.name`, giấy phép lấy từ trang dataset, và chọn `messages_field` (cột hội thoại, hiểu cả dạng role/content lẫn ShareGPT from/value) hoặc `mapping.input` / `mapping.expected_output` (tùy chọn thêm `mapping.context`, `mapping.reasoning`).

```bash
python -m local_ai.data hf-sft --config configs/datasets/hf_sft.json [--dataset org/name] [--limit 1000]
```

Kết quả nằm trong `output_dir`:
- `sft.jsonl`: dùng để train;
- `train.jsonl`;
- `rejected.jsonl`: dòng bị loại kèm lý do;
- `manifest.json`.

Trong `sft.jsonl`:
- hội thoại nhiều lượt giữ nguyên mọi lượt, kể cả system;
- `context` được đặt trước câu hỏi;
- `reasoning` được đưa vào khối `<think>...</think>` trước câu trả lời (định dạng suy luận của Qwen3).

## Bước 2: fine-tune
`configs/training/sft.json` chỉ định model gốc, đường dẫn `sft.jsonl`, thư mục đầu ra và siêu tham số; `method` là `full` hoặc `lora`; `quantization` (tùy chọn) ghi đè kiểu nén của model gốc.

```bash
python -m local_ai.training.finetune --config configs/training/sft.json --dry-run
python -m local_ai.training.finetune --config configs/training/sft.json
```

QLoRA (LoRA trên model gốc nén 4bit) cho model chính dùng cấu hình mẫu `configs/training/qlora_primary.json` (`method: "lora"` + `quantization: "4bit"`); có thể ghi đè bằng `--quantization 4bit|8bit|none`. Full fine-tune không chạy trên model nén.

```bash
python -m local_ai.training.finetune --config configs/training/qlora_primary.json --dry-run
```

Các khóa hữu ích khác trong cấu hình huấn luyện:
- `max_steps`: dừng sau đúng số bước này, ví dụ `2` để chạy thử;
- `require_gpu: false`: cho phép chạy trên CPU.

Để chấm model sau khi train, thêm một mục vào danh sách model, trỏ `adapter_path` tới thư mục adapter vừa lưu. Ví dụ `{"name": "smoke-lora", "source": "Qwen/Qwen2.5-0.5B-Instruct", "adapter_path": ".runs/sft/adapter", ...}`. Sau đó chạy `python -m local_ai.evaluation --model smoke-lora`. `max_new_tokens` (mặc định 512) giới hạn độ dài câu trả lời.

Checkpoint lưu thành `checkpoint-N`; chạy lại sẽ tiếp tục từ checkpoint mới nhất (`--no-resume` để làm lại). Thiếu GPU hoặc thư viện (kể cả bitsandbytes khi dùng QLoRA) thì lệnh báo `"status": "skipped"`.

### Chạy thử cả chuỗi trên CPU (không cần GPU, không tải gì)
Test `tests/test_m6_cpu_pipeline.py` tự tạo tokenizer (BPE, có `chat_template`) và model Llama tí hon (2 lớp, hidden 64, khởi tạo ngẫu nhiên). Sau đó test chạy thật cả chuỗi:
1. fixture preset → `sft.jsonl`;
2. LoRA 2 bước trên CPU;
3. lưu adapter;
4. nạp lại model gốc cùng adapter;
5. chạy eval 30 câu.

Test chạy với `HF_HUB_OFFLINE=1`. Trên máy 4 CPU, test mất khoảng 6 giây. Máy thiếu torch, transformers, trl, peft hoặc datasets thì test tự bỏ qua.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers trl peft datasets
python -m unittest tests.test_m6_cpu_pipeline -v
```

## Bước 3: đánh giá (eval)
Bộ câu hỏi `data/eval/eval_v1.jsonl` có 30 câu:
- chia theo ngôn ngữ: 15 câu tiếng Việt, 15 câu tiếng Anh;
- chia theo nhóm: 8 câu `code`, 14 câu `reasoning`, 8 câu `tool_use` (trả JSON gọi công cụ).

Mỗi câu ghi `language`, `group` và cách chấm `scoring`:

| scoring | Cách chấm |
| --- | --- |
| `exact` | Trả lời đúng bằng `expected`. Không phân biệt hoa thường, gộp khoảng trắng, chuẩn hóa Unicode tiếng Việt. |
| `contains` | Câu trả lời chứa `expected`, chuẩn hóa như trên. |
| `regex` | Câu trả lời khớp `pattern`. Không phân biệt hoa thường. |
| `python_tests` | Lấy code trong câu trả lời, chạy cùng `tests` trong `PythonSandbox`, giới hạn `timeout_s` giây. Code sai cú pháp, lỗi khi chạy hoặc lặp vô hạn đều tính là không đạt. |

Trước khi chấm, khối `<think>...</think>` trong câu trả lời được bỏ đi. Mỗi câu có `reference` là một câu trả lời đúng mẫu; test kiểm tra mọi `reference` đều đạt.

```bash
python -m local_ai.evaluation --scripted                  # chạy thử bằng đáp án mẫu (không cần model), kết quả 30/30
python -m local_ai.evaluation --model ollama              # chấm một model trong configs/models/platform.json
python -m local_ai.evaluation --model smoke --train-data data/processed/hf_mix/sft.jsonl
```

Báo cáo ghi vào `.runs/eval/<model>-<thời điểm>/`, hoặc thư mục ghi ở `--output`:
- `report.json`: kết quả từng câu;
- `report.md`: tỉ lệ đạt theo nhóm và theo ngôn ngữ, cùng danh sách câu không đạt kèm lý do.

Nếu model lỗi (ví dụ server chưa chạy), lệnh dừng và ghi `status: "error"`. Model chạy bằng transformers mà máy thiếu torch hoặc transformers thì lệnh báo bỏ qua.

**Chặn trùng train/eval** theo id, theo nội dung, và theo câu hỏi đã chuẩn hóa (kể cả các lượt user trong hội thoại):
- Bước build dữ liệu (`hf-sft`, `build`) dừng và báo lỗi nếu dữ liệu train trùng với file trong `eval_sources`. Các cấu hình có sẵn đã chặn cả `seed_eval.jsonl` lẫn `eval_v1.jsonl`.
- Lệnh eval với `--train-data <train.jsonl hoặc sft.jsonl>` từ chối chạy nếu dữ liệu train chứa câu eval.

`PythonSandbox` chỉ chạy code trong một tiến trình riêng, có giới hạn thời gian. Nó không phải lớp cách ly an toàn. Nếu chấm code của model lạ, hãy chạy trong container.

## Lộ trình
Theo `TASKS.md`: M1 agent chịu lỗi và dọn repo → M2 model ảnh+chữ, nén 4-bit (QLoRA), ước tính VRAM → M3 adapter server local (Ollama, llama.cpp) → M4 preset dataset → M5 eval mở rộng → M6 test chạy thật trên CPU → M7 tổng kết. Tiến độ từng mốc ghi trong bảng ở `TASKS.md`.
