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
- `local_ai/evaluation`: chấm điểm model.
- `local_ai/agents`, `local_ai/tools`: agent có giới hạn vòng lặp và các công cụ (dùng để thử model); công cụ lỗi không làm sập agent.
- `archive/`: phần đã cất, không còn dùng (corpus 10T token, hướng dẫn nanoGPT cũ).

## Cài đặt và kiểm tra
```bash
python -m pip install datasets                        # tải dataset Hugging Face
python -m pip install torch transformers trl peft     # huấn luyện (cần GPU)
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
Sửa `configs/datasets/hf_sft.json`: điền `hf_dataset.name`, giấy phép lấy từ trang dataset, và chọn `messages_field` (cột hội thoại) hoặc `mapping.input` / `mapping.expected_output`.

```bash
python -m local_ai.data hf-sft --config configs/datasets/hf_sft.json [--dataset org/name] [--limit 1000]
```

Kết quả nằm trong `output_dir`: `sft.jsonl` (dùng để train), `train.jsonl`, `rejected.jsonl`, `manifest.json`.

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

Checkpoint lưu thành `checkpoint-N`; chạy lại sẽ tiếp tục từ checkpoint mới nhất (`--no-resume` để làm lại). Thiếu GPU hoặc thư viện (kể cả bitsandbytes khi dùng QLoRA) thì lệnh báo `"status": "skipped"`.

## Lộ trình
Theo `TASKS.md`: M1 agent chịu lỗi và dọn repo → M2 model ảnh+chữ, nén 4-bit (QLoRA), ước tính VRAM → M3 adapter server local (Ollama, llama.cpp) → M4 preset dataset → M5 eval mở rộng → M6 test chạy thật trên CPU → M7 tổng kết. Tiến độ từng mốc ghi trong bảng ở `TASKS.md`.
