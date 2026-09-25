# Fine-tune model Qwen3 (Huihui) chạy cục bộ

Repo dùng để fine-tune model có sẵn trên Hugging Face, mọi bước đều điều khiển bằng file cấu hình:

```
dataset Hugging Face → kiểm tra, loại trùng, lọc chất lượng, chặn rò rỉ eval → sft.jsonl → fine-tune LoRA/QLoRA → đánh giá
```

Model chính là `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (ảnh + chữ, 27,78 tỷ tham số); các model khác nằm trong `configs/models/platform.json`. Repo không tự tải model: test chạy được mà không cần mạng hay GPU.

Kế hoạch tuần 1–2 (M1–M14) và bằng chứng từng mốc: `TASKS.md`. Tiến độ, việc chủ repo tự làm và lỗi còn tồn: `memory.md`.

Chạy trên Colab miễn phí, không cần máy có GPU:
- train model nhỏ: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/train_colab.ipynb) (hướng dẫn trên iPhone: `docs/TRAIN_COLAB.md`);
- agent với model thật (Ollama): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/agent_colab.ipynb).

## Trạng thái sau tuần 1 và tuần 2
| Phần | Làm được gì | Đã kiểm chứng thế nào |
| --- | --- | --- |
| Dữ liệu | 3 preset Hugging Face, trộn theo tỉ lệ; giữ hội thoại nhiều lượt, reasoning, context; loại trùng; chặn trùng với eval. **Tuần 2:** bộ lọc chất lượng (ngôn ngữ ưu tiên tiếng Việt, độ dài, lặp, gần trùng bằng MinHash), thống kê trước/sau lọc trong `manifest.json` | Test bằng fixture. **Chạy thật** qua mạng: trộn 1000/750/750 dòng, giữ 2500/2500, khoảng 15 giây; lọc 2000 dòng thật mất khoảng 5 giây, loại 1 dòng |
| Fine-tune | Full, LoRA, QLoRA (4bit); model chữ và model ảnh + chữ; GPU không bf16 thì dùng fp16. **Tuần 2:** đẩy checkpoint lên Hugging Face và tự train tiếp khi Colab ngắt; ước tính thời gian trên T4 | **Chạy thật LoRA trên CPU** với model tí hon (khoảng 6 giây). QLoRA, model ảnh + chữ và phần đẩy lên Hugging Face **mới test bằng module giả**, chưa chạy trên GPU |
| Eval | 30 câu Việt + Anh, 4 cách chấm, báo cáo JSON + Markdown. **Tuần 2:** bảng so sánh trước/sau khi train | Test đủ 4 cách chấm; **chạy thật** với model tí hon sau khi train |
| Model qua server local | Ollama, llama.cpp, vLLM (chuẩn OpenAI); mục `ollama-colab` (`qwen3:4b`) | Test bằng server HTTP giả; **chưa chạy với server thật** |
| Agent | Công cụ lỗi hoặc model lỗi không làm sập agent; tách JSON từ câu trả lời lộn xộn. **Tuần 2:** prompt gửi danh sách công cụ; 5 nhiệm vụ mẫu với calculator và TerminalTool | Test bằng model giả và server OpenAI giả (5/5 nhiệm vụ); **chưa chạy với model thật** |
| Runtime gộp từ repo Agent (tuần 2) | `TerminalTool` (allowlist, tắt mặc định, không qua shell, có log); gateway hàng đợi job (tắt mặc định, chỉ `127.0.0.1`, bắt buộc token) | Test các kiểu chèn lệnh, tham số nguy hiểm, đường dẫn ra ngoài thư mục làm việc; lệnh chạy thật trong thư mục tạm |
| Notebook Colab (tuần 2) | `train_colab` (smoke hoặc light, QLoRA fp16, train tiếp khi Colab ngắt); `agent_colab` (Ollama + `qwen3:4b`); hướng dẫn trên iPhone | Hợp lệ theo nbformat; mọi lệnh của notebook chạy được bằng `--dry-run`. **Chưa chạy trên Colab thật** |

## Cấu trúc
- `local_ai/data`: đọc dữ liệu, kiểm tra schema, loại trùng, chặn rò rỉ eval, xuất `sft.jsonl`; bước tải dataset Hugging Face (`hub.py`); bộ lọc chất lượng (`quality.py`); quét khóa bí mật (`secrets.py`).
- `local_ai/training`: khung fine-tune SFT (`finetune.py`: full, LoRA hoặc QLoRA); đẩy checkpoint và adapter lên Hugging Face Hub (`hub.py`); ước tính thời gian train trên Colab (`estimate.py`).
- `local_ai/models`: cấu hình model, adapter chạy model Hugging Face (chữ hoặc ảnh + chữ, nén 4bit/8bit), adapter gọi server local kiểu OpenAI (`openai_compatible.py`), router chọn model theo khả năng, ước tính VRAM (`vram.py`).
- `local_ai/evaluation`: bộ eval (`suite.py`) với 4 cách chấm, lệnh `python -m local_ai.evaluation`; câu hỏi nằm trong `data/eval/eval_v1.jsonl`; so sánh 2 báo cáo (`compare.py`).
- `local_ai/agents`, `local_ai/tools`: agent có giới hạn vòng lặp và các công cụ (dùng để thử model); công cụ lỗi không làm sập agent; 5 nhiệm vụ mẫu cho agent (`agents/tasks.py`).
- `local_ai/runtime`: runtime chạy việc gộp từ repo Agent: chạy lệnh dạng list không qua shell, `TerminalTool` cho agent (allowlist, tắt mặc định), hàng đợi job và gateway HTTP (tắt mặc định). Chi tiết gộp và rủi ro bảo mật: `docs/GOP_AGENT.md`.
- `local_ai/config`, `local_ai/experiments`, `local_ai/memory`: nạp danh sách model, ghi lại lượt chạy (`RunTracker`), bộ nhớ hội thoại ngắn.
- `configs/`: mọi file cấu hình (model, dataset, preset, huấn luyện). `data/`: dữ liệu mẫu (`data/raw/`) và bộ eval (`data/eval/`). `tests/`: test theo từng mốc. `docs/ARCHITECTURE.md`: kiến trúc. `docs/TRAIN_COLAB.md`: train trên Colab bằng iPhone.
- `notebooks/`: notebook Colab, sinh từ `notebooks/build.py` (lưu không kèm output).
- `archive/`: phần đã cất, không còn dùng (corpus 10T token, hướng dẫn nanoGPT cũ, code gốc của repo Agent, nhật ký tuần 1).

## Cài đặt và kiểm tra
Chạy mọi lệnh trong README từ thư mục gốc của repo, vì các đường dẫn trong file cấu hình (`dataset_path`, `output_dir`...) tính từ đó.

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

Phiên bản đã chạy thử: Python 3.11, torch 2.14.0 (bản CPU), transformers 5.17.0, trl 1.13.0, peft 0.21.0, datasets 5.0.1. Model chính cần transformers bản 5.x.

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

Cần cài Ollama, llama.cpp hoặc vLLM trước. Các lệnh `ollama`, `llama-server`, `vllm` dưới đây **chưa chạy thử** trong môi trường phát triển của repo, vì cần cài phần mềm và tải model. Phần của repo gọi tới server đã được test bằng server giả.

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

## Công cụ chạy lệnh cho agent (TerminalTool) và gateway
`TerminalTool` (`local_ai/runtime/terminal.py`) cho agent chạy lệnh thật trên máy, nhưng **tắt sẵn**. Cấu hình ở `configs/tools/terminal.json`:
- chỉ các lệnh có trong `allowed_commands` được chạy: `pwd`, `ls`, `cat`, `head`, `tail`, `grep`, `find`, `git status`/`git log`, `python --version`, `uname`, `whoami`, `date`...;
- lệnh chạy dạng list, **không qua shell**, trong thư mục `workspace`, có `timeout_s`;
- mọi lệnh, kể cả lệnh bị từ chối, được ghi vào `log_path` (JSONL).

Các trường hợp bị từ chối:
- lệnh có ký tự điều khiển shell (`;` `&&` `|` `$()` backtick `<` `>`, xuống dòng);
- lệnh ngoài allowlist;
- tham số nguy hiểm (`find -exec`, `find -delete`, `git log --output`, `tail -f`...);
- đường dẫn ra ngoài thư mục làm việc (`/etc/passwd`, `../..`).

Muốn bật: tạo bản sao của file cấu hình với `"enabled": true`, hoặc bật trong code:
```python
from local_ai.runtime.terminal import TerminalTool, register_terminal
register_terminal(registry, TerminalTool(enabled=True))   # agent gọi {"tool": "terminal", "arguments": {"command": "ls"}}
```

Gateway hàng đợi job (`python -m local_ai.runtime.gateway`) cũng **tắt sẵn** (`configs/runtime/gateway.json`):
- chỉ nghe `127.0.0.1`;
- bắt buộc token dài ít nhất 16 ký tự, đọc từ biến môi trường `LOCAL_AI_GATEWAY_TOKEN`;
- không có endpoint nào chạy lệnh qua mạng.

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
python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --output data/processed/hf_sft
```

Trộn nhiều preset (lặp lại `--preset tên:tỉ_lệ`) thì kết quả chung vào một `sft.jsonl`, mặc định ở `data/processed/hf_sft/`, đúng chỗ các cấu hình train đọc:
- Tỉ lệ được chuẩn hóa về tổng 1.
- Nếu không ghi `--total`, lệnh lấy tổng số dòng lớn nhất sao cho không preset nào vượt `limit` của nó.
- `--limit 20` thì mỗi preset chỉ đọc tối đa 20 dòng, hợp để chạy thử.
- Tỉ lệ tính trên số dòng đọc vào. Dòng bị loại (trùng, sai dạng, không qua bộ lọc chất lượng) làm tỉ lệ cuối lệch nhẹ; số dòng thật của từng nguồn ghi trong `manifest.json`.

### Bộ lọc chất lượng
`hf-sft` lọc dữ liệu theo `configs/datasets/quality.json` (tắt bằng `--no-quality`; lệnh `build` bật bằng `--quality`). Các bộ lọc chạy sau bước kiểm tra schema và loại trùng tuyệt đối, theo thứ tự:

| Bộ lọc | Loại dòng nào |
| --- | --- |
| `language` (ưu tiên tiếng Việt) | Nhận tiếng Việt trước tiên, theo chữ có dấu, nên câu trộn tiếng Việt với code hay tiếng Anh vẫn tính là tiếng Việt. Loại: chữ không phải Latin (ví dụ tiếng Trung), tiếng Việt không dấu, và dòng có nội dung khác ngôn ngữ nguồn khai báo (preset `vietnamese` phải là tiếng Việt có dấu). Dòng tiếng Việt có dấu trong nguồn khác (ví dụ preset `code`) vẫn được giữ. |
| `length` | Câu hỏi dưới 3 ký tự, câu trả lời dưới 2 ký tự, hoặc cả hội thoại quá 16.000 ký tự. |
| `repetition` | Xét từng lượt riêng; bỏ qua khối code và lệnh LaTeX vì chúng lặp là bình thường. Loại lượt có một từ lặp liền quá 8 lần, hoặc (với lượt từ 50 từ) hơn 50% số câu hay cụm 5 từ là bản lặp. |
| `near_duplicate` | Gần trùng: Jaccard trên cụm 3 từ từ 0,8 trở lên, tìm bằng MinHash tự viết (128 ngăn, LSH 32 dải) rồi so lại bằng Jaccard thật. Giữ dòng có id đứng trước. |

Dòng bị loại nằm trong `rejected.jsonl`, kèm `quality_filter` (tên bộ lọc) và `quality_reason` (lý do). Mục `quality` của `manifest.json` ghi:
- số dòng, ngôn ngữ, tỉ lệ tiếng Việt và độ dài, **trước và sau** khi lọc;
- số dòng mỗi bộ lọc đã loại;
- cấu hình đã dùng.

Thử trên 2000 dòng thật của 3 preset (25/9/2026): bộ lọc chạy khoảng 5 giây và chỉ loại 1 dòng (lời giải toán lặp công thức). 3 dataset này đã được làm sạch sẵn; bộ lọc chủ yếu để chặn dữ liệu bẩn khi thêm dataset mới.

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
python -m local_ai.evaluation --scripted --train-data data/processed/hf_sft/sft.jsonl   # kiểm tra dữ liệu train không chứa câu eval
```

Báo cáo ghi vào `.runs/eval/<model>-<thời điểm>/`, hoặc thư mục ghi ở `--output`:
- `report.json`: kết quả từng câu;
- `report.md`: tỉ lệ đạt theo nhóm và theo ngôn ngữ, cùng danh sách câu không đạt kèm lý do.

So sánh 2 lần chấm (ví dụ trước và sau khi train): `python -m local_ai.evaluation.compare <trước>/report.json <sau>/report.json`. Lệnh in bảng tỉ lệ đạt theo nhóm và theo ngôn ngữ, mức thay đổi (điểm %), cùng các câu mới đạt và mới trượt. `--max-new-tokens` ghi đè độ dài câu trả lời tối đa để chấm nhanh hơn; `--dry-run` chỉ in kế hoạch, không nạp model.

Nhóm câu eval (`code`, `reasoning`, `tool_use`) là cách chia riêng của bộ eval, khác với `domain` của dữ liệu train (`coding`, `math_logic`, `chat`...).

Nếu model lỗi (ví dụ server chưa chạy), lệnh dừng và ghi `status: "error"`. Model chạy bằng transformers mà máy thiếu torch hoặc transformers thì lệnh báo bỏ qua.

**Chặn trùng train/eval** theo id, theo nội dung, và theo câu hỏi đã chuẩn hóa (kể cả các lượt user trong hội thoại):
- Bước build dữ liệu (`hf-sft`, `build`) dừng và báo lỗi nếu dữ liệu train trùng với file trong `eval_sources`. Các cấu hình có sẵn đã chặn cả `seed_eval.jsonl` lẫn `eval_v1.jsonl`.
- Lệnh eval với `--train-data <train.jsonl hoặc sft.jsonl>` từ chối chạy nếu dữ liệu train chứa câu eval.

`PythonSandbox` chỉ chạy code trong một tiến trình riêng, có giới hạn thời gian. Nó không phải lớp cách ly an toàn. Nếu chấm code của model lạ, hãy chạy trong container.

## Train trên GPU: smoke → light → primary
Làm lần lượt từ model nhỏ đến model lớn; mỗi bước chạy `--dry-run` trước để xem cấu hình. **Các lệnh train trên GPU chưa chạy thử** (môi trường phát triển không có GPU); lệnh đã chạy được trên CPU với model tí hon (xem trên).

Chuẩn bị:
- trên máy GPU, cài torch bản CUDA, rồi `python -m pip install transformers trl peft datasets bitsandbytes`;
- tạo dữ liệu vào đúng đường dẫn mà các cấu hình train đang đọc:

```bash
python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --output data/processed/hf_sft
```

| Bước | Model | Lệnh train | VRAM ước tính | GPU gợi ý |
| --- | --- | --- | ---: | --- |
| 1. smoke | `smoke` (Qwen2.5-0.5B), LoRA bf16 | `python -m local_ai.training.finetune --config configs/training/sft.json` | 2,2 GB | GPU bất kỳ từ 4 GB (T4 tự dùng fp16) |
| 2. light | `light` (Qwen3-4B), LoRA bf16 | `python -m local_ai.training.finetune --config configs/training/sft.json --base-model light --output-dir .runs/sft_light` | 11,0 GB | T4 16 GB, RTX 3060 12 GB; thêm `--quantization 4bit` (QLoRA) thì còn 3,8 GB |
| 3. primary | `primary` (27,78 tỷ), QLoRA 4bit | `python -m local_ai.training.finetune --config configs/training/qlora_primary.json` | 20,5 GB | 24 GB (RTX 3090/4090, L4, A10); LoRA bf16 không nén cần khoảng 70,5 GB (A100 80 GB) |

Chấm sau mỗi bước: `smoke-lora`, `light-lora`, `primary-qlora` trong `configs/models/platform.json` đã trỏ sẵn `adapter_path` tới thư mục adapter của từng bước. So với model gốc bằng cùng một bộ eval:
```bash
python -m local_ai.evaluation --model smoke --train-data data/processed/hf_sft/sft.jsonl
python -m local_ai.evaluation --model smoke-lora --train-data data/processed/hf_sft/sft.jsonl
python -m local_ai.evaluation --model primary-qlora
```

Số VRAM lấy từ `python -m local_ai.models.vram` (batch 1, khoảng 2048 token, bật gradient checkpointing). Đây chỉ là ước lượng; khi chạy thật, hãy đo lại và sửa hệ số trong `local_ai/models/vram.py`. Chạy `light` sau khi xong `smoke` thì luôn ghi `--output-dir` khác, để không chạy tiếp nhầm checkpoint của model khác.

## Train trên Colab miễn phí
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/train_colab.ipynb)

Notebook `notebooks/train_colab.ipynb` train trên GPU T4 miễn phí của Colab. **Hướng dẫn từng bước trên iPhone: `docs/TRAIN_COLAB.md`.**

Ô đầu tiên chọn model:

| Model | Cấu hình train | Chấm sau khi train | Thời gian ước tính (train + 2 lần chấm) |
| --- | --- | --- | --- |
| `smoke` (Qwen2.5-0.5B) | `configs/training/colab_smoke.json`: QLoRA 4bit, fp16, batch 4, `max_length` 1024 | `smoke-colab` | khoảng 21 phút |
| `light` (Qwen3-4B) | `configs/training/colab_light.json`: QLoRA 4bit, fp16, batch 1 (tích lũy 16), `max_length` 2048, VRAM ước tính 3,8 GB | `light-colab` | khoảng 109 phút |

Các ô sau chạy lần lượt:
1. kiểm tra GPU; chưa có thì hướng dẫn chọn T4;
2. tải repo, cài thư viện đã ghim phiên bản (không cài lại torch);
3. đọc `HF_TOKEN` từ Colab Secrets;
4. lấy 2000 dòng từ 3 preset;
5. in ước tính thời gian, và số bước đã train nếu Colab từng ngắt;
6. chấm model gốc;
7. train QLoRA;
8. chấm lại model sau khi train;
9. in bảng so sánh trước/sau;
10. đẩy adapter lên repo Hugging Face riêng tư.

**Cần chuẩn bị:** token Hugging Face quyền Write, lưu trong Colab Secrets (biểu tượng chìa khóa) với tên `HF_TOKEN`, bật Notebook access. Token không nằm trong notebook hay repo.

**Colab ngắt giữa chừng:** mở lại notebook, giữ nguyên model đã chọn và chạy Run all.
- Cờ `--push-to-hub` của lệnh train đẩy checkpoint mới nhất vào thư mục `last-checkpoint` của repo riêng tư sau mỗi `save_steps` bước (`light`: 10, `smoke`: 25).
- Lần chạy sau, lệnh train tự tải checkpoint đó về `.runs/colab_<model>/_hub/last-checkpoint` rồi train tiếp.

Bước cuối của notebook đẩy adapter lên repo riêng tư bằng lệnh con `push-adapter` của `local_ai.training.hub`. Token đọc từ biến môi trường `HF_TOKEN`; thêm `--dry-run` để chỉ kiểm tra tham số, không gọi mạng.

**Ước tính thời gian** (`python -m local_ai.training.estimate`) không tải model, chỉ dựa vào cấu hình và `sft.jsonl`:
- số token mỗi dòng, kể cả phần đệm khi batch lớn hơn 1;
- thông lượng giả định của T4.

Đây là ước lượng thô, chưa đo trên T4 thật; cách tính ghi ở đầu `local_ai/training/estimate.py`.

Nút Colab mở notebook ở nhánh `main`, nên chỉ dùng được sau khi gộp PR tuần 2. **Notebook chưa chạy thử trên Colab thật.** Test `tests/test_m10_colab.py` và `tests/test_m11_colab_light.py` chỉ kiểm tra notebook hợp lệ và chạy các lệnh của nó bằng `--dry-run` với cả 2 model, ví dụ:

```bash
python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --total 2000 --output data/processed/hf_sft --dry-run
python -m local_ai.training.estimate --config configs/training/colab_light.json --max-new-tokens 512 --dry-run
python -m local_ai.training.finetune --config configs/training/colab_light.json --push-to-hub --hub-model-id ten-ban/huyen-light-qlora --dry-run
python -m local_ai.evaluation --model light-colab --max-new-tokens 512 --dry-run
python -m local_ai.evaluation.compare .runs/eval/light/truoc/report.json .runs/eval/light/sau/report.json --dry-run
```

Muốn sửa notebook thì sửa nội dung ô trong `notebooks/build.py`, rồi chạy `python notebooks/build.py`; test báo lỗi nếu file `.ipynb` không khớp.

## Agent chạy model thật trên Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/agent_colab.ipynb)

Notebook `notebooks/agent_colab.ipynb` chạy theo các bước:
1. cài Ollama bản đã ghim (0.34.4);
2. tải model `qwen3:4b` (khoảng 2,5 GB);
3. cho agent của repo làm 5 nhiệm vụ mẫu, gọi model qua adapter kiểu OpenAI (mục `ollama-colab` trong danh sách model).

Notebook không cần token và không cần cài gói pip nào.

Các nhiệm vụ nằm trong `data/eval/agent_tasks_v1.jsonl`, dùng 2 công cụ:
- `calculator`;
- `TerminalTool`: chỉ bật trong thư mục làm việc riêng của từng nhiệm vụ, chép từ `data/eval/agent_workspace/`.

| Nhiệm vụ | Công cụ phải dùng |
| --- | --- |
| Tính (17 * 23) + 158 | calculator |
| Tính tiền 3 áo giá 125.000 đồng, giảm 10% | calculator |
| Liệt kê file trong thư mục làm việc | terminal (`ls`) |
| Đọc mã số đơn hàng trong `ghi_chu.txt` | terminal (`cat`) |
| Tính tổng cột `so_luong` của `du_lieu.csv` | terminal, rồi calculator |

Một nhiệm vụ chỉ tính là đạt khi câu trả lời đúng **và** agent đã thật sự gọi mọi công cụ cần dùng, để không tính trường hợp model đoán mò. Lệnh in trace từng nhiệm vụ (kế hoạch, công cụ đã gọi, kết quả) và tỉ lệ thành công ở cuối:

```bash
python -m local_ai.agents.tasks --scripted                   # dùng câu trả lời mẫu, không cần model: 5/5
python -m local_ai.agents.tasks --model ollama-colab --dry-run
python -m local_ai.agents.tasks --model ollama-colab --output .runs/agent_tasks/report.json   # cần Ollama đang chạy
```

Để model thật biết gọi công cụ nào, agent gửi kèm trong prompt danh sách công cụ, cách truyền tham số và các kết quả trước đó.

**Chưa chạy với Ollama thật** (môi trường phát triển không tải model). Test `tests/test_m13_agent_colab.py` gồm:
- kiểm tra notebook hợp lệ và chạy lệnh của nó bằng `--dry-run`;
- chạy 5 nhiệm vụ với công cụ thật, qua một server HTTP giả nói chuẩn OpenAI.

## Lộ trình tiếp theo (đề xuất, chưa làm)
Tuần 1 đề xuất 6 việc. Tuần 2 đã làm phần chuẩn bị cho việc chạy thật trên GPU và thử Ollama (notebook Colab train và agent), nhưng chưa chạy thật. Các việc còn lại gom vào 7 mốc đề xuất cho tuần 3 dưới đây. Đây chỉ là đề xuất, chủ repo chọn việc nào thì mới đưa vào `TASKS.md`:
1. **M15 – Số đo thật trên Colab.** Dựa trên kết quả chủ repo chạy `train_colab` (smoke, light) và `agent_colab`: sửa hằng số ước tính trong `local_ai/training/estimate.py` và `local_ai/models/vram.py`, rồi ghi bảng số đo thật (thời gian, VRAM, điểm trước/sau, tỉ lệ agent) vào README.
2. **M16 – Notebook bền hơn:**
   - dừng Run all khi một lệnh `!python` lỗi;
   - lưu báo cáo chấm trước lên repo Hugging Face, để chạy lại sau khi Colab ngắt không phải chấm lại.
3. **M17 – Model chính (ảnh + chữ):** chỉ gắn LoRA vào phần ngôn ngữ, không gắn vào phần xử lý ảnh; đổi `torch_dtype` sang `dtype` theo transformers 5.x.
4. **M18 – Dùng model đã train trong agent:** gộp adapter vào model gốc, xuất GGUF để chạy bằng Ollama, rồi cho agent làm 5 nhiệm vụ mẫu với model vừa train, so với model gốc.
5. **M19 – Mở rộng eval:** thêm nhiệm vụ agent nhiều bước (10 nhiệm vụ) và câu dùng công cụ; tự chấm ngay sau mỗi lần train.
6. **M20 – Dữ liệu:**
   - chặn gần trùng giữa dữ liệu train và eval bằng MinHash của M12 (hiện chỉ chặn trùng chính xác và trùng câu hỏi);
   - thêm một preset tiếng Việt nữa, sau khi đọc kỹ giấy phép.
7. **M21 – Tổng kết tuần 3.**

Việc cần chủ repo quyết định: PR #4 (đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B).

Kế hoạch tuần 1–2 (M1–M14) và bằng chứng từng mốc nằm trong `TASKS.md`.
