# Fine-tune model Qwen3 (Huihui) chạy cục bộ

Repo dùng để fine-tune model có sẵn trên Hugging Face, mọi bước đều điều khiển bằng file cấu hình:

```
dataset Hugging Face → kiểm tra, loại trùng, lọc chất lượng, chặn rò rỉ eval → sft.jsonl → fine-tune LoRA/QLoRA → đánh giá
```

Model chính là `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (ảnh + chữ, 27,78 tỷ tham số); các model khác nằm trong `configs/models/platform.json`. Repo không tự tải model: test chạy được mà không cần mạng hay GPU.

Kế hoạch tuần 1–3 và bằng chứng từng mốc: `TASKS.md` (tuần 3) và `archive/tasks-tuan-1-2.md` (tuần 1–2). Tiến độ, việc chủ repo tự làm và lỗi còn tồn: `memory.md`.

Chạy trên Colab miễn phí, không cần máy có GPU:
- train model nhỏ: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/train_colab.ipynb) (hướng dẫn trên iPhone: `docs/TRAIN_COLAB.md`);
- agent với model thật (Ollama): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/agent_colab.ipynb).

## Trạng thái hiện tại
Cập nhật 26/9/2026, sau khi chạy thật trên Colab T4 (M15).

| Phần | Làm được gì | Đã kiểm chứng thế nào |
| --- | --- | --- |
| Dữ liệu | 3 preset Hugging Face, trộn theo tỉ lệ; giữ hội thoại nhiều lượt, reasoning, context; loại trùng; chặn trùng với eval. **Tuần 2:** bộ lọc chất lượng (ngôn ngữ ưu tiên tiếng Việt, độ dài, lặp, gần trùng bằng MinHash), thống kê trước/sau lọc trong `manifest.json` | Test bằng fixture. **Chạy thật** qua mạng: trộn 1000/750/750 dòng, giữ 2500/2500, khoảng 15 giây; lọc 2000 dòng thật mất khoảng 5 giây, loại 1 dòng. Notebook Colab lấy 2000 dòng thật mỗi lần chạy |
| Fine-tune | Full, LoRA, QLoRA (4bit); model chữ và model ảnh + chữ (LoRA chỉ gắn phần ngôn ngữ, M17); GPU không bf16 thì dùng fp16. Đẩy checkpoint lên Hugging Face và tự train tiếp khi Colab ngắt; ước tính thời gian và VRAM trên T4 theo số đo thật | **Chạy thật trên Colab T4** (25–26/9): QLoRA `smoke` và `light`, mỗi model 125 bước; `light` có một lần Colab ngắt và train tiếp được. LoRA chạy thật trên CPU với model tí hon (cả model ảnh + chữ Qwen3.5 tí hon). **Chưa chạy trên Colab thật:** model chính 27B (cần GPU 40–48 GB) |
| Eval | 30 câu Việt + Anh, 4 cách chấm, báo cáo JSON + Markdown; bảng so sánh trước/sau khi train | **Chạy thật trên Colab T4**: `smoke` 16 → 14/30; `light` 17 → 19/30 nhưng tool_use 7/8 → 3/8 (sửa ở M19, M20) |
| Model qua server local | Ollama, llama.cpp, vLLM (chuẩn OpenAI); mục `ollama-colab` (`qwen3:4b`) | **Chạy thật** Ollama + `qwen3:4b` trên Colab T4 (26/9). llama.cpp và vLLM mới test bằng server HTTP giả |
| Agent | Công cụ lỗi hoặc model lỗi không làm sập agent; tách JSON từ câu trả lời lộn xộn; prompt gửi danh sách công cụ; 5 nhiệm vụ mẫu với calculator và TerminalTool | **Chạy thật** với `qwen3:4b` trên Colab T4: 4/5 nhiệm vụ trong 11,1 phút. Test bằng model giả và server OpenAI giả (5/5 nhiệm vụ) |
| Runtime gộp từ repo Agent (tuần 2) | `TerminalTool` (allowlist, tắt mặc định, không qua shell, có log); gateway hàng đợi job (tắt mặc định, chỉ `127.0.0.1`, bắt buộc token) | Test các kiểu chèn lệnh, tham số nguy hiểm, đường dẫn ra ngoài thư mục làm việc; lệnh chạy thật trong thư mục tạm; trên Colab, TerminalTool chạy thật với agent |
| Notebook Colab (tuần 2) | `train_colab` (smoke hoặc light, QLoRA fp16, train tiếp khi Colab ngắt); `agent_colab` (Ollama + `qwen3:4b`); hướng dẫn trên iPhone. **Tuần 3:** lệnh lỗi thì dừng Run all, dùng lại báo cáo chấm trước, tự gỡ torchao, ghi số đo thật | **Chạy thật trên Colab T4** (25–26/9): `train_colab` với `smoke` và `light`, `agent_colab`. Hợp lệ theo nbformat; mọi lệnh của notebook chạy được bằng `--dry-run` |

## Cấu trúc
- `local_ai/data`: đọc dữ liệu, kiểm tra schema, loại trùng, chặn rò rỉ eval, xuất `sft.jsonl`; bước tải dataset Hugging Face (`hub.py`); bộ lọc chất lượng (`quality.py`); quét khóa bí mật (`secrets.py`).
- `local_ai/training`: khung fine-tune SFT (`finetune.py`: full, LoRA hoặc QLoRA); đẩy checkpoint và adapter lên Hugging Face Hub (`hub.py`); ước tính thời gian train trên Colab (`estimate.py`); so số đo thật với ước tính (`calibrate.py`).
- `local_ai/models`: cấu hình model, adapter chạy model Hugging Face (chữ hoặc ảnh + chữ, nén 4bit/8bit), adapter gọi server local kiểu OpenAI (`openai_compatible.py`), router chọn model theo khả năng, ước tính VRAM (`vram.py`).
- `local_ai/evaluation`: bộ eval (`suite.py`) với 4 cách chấm, lệnh `python -m local_ai.evaluation`; câu hỏi nằm trong `data/eval/eval_v1.jsonl`; so sánh 2 báo cáo (`compare.py`).
- `local_ai/agents`, `local_ai/tools`: agent có giới hạn vòng lặp và các công cụ (dùng để thử model); công cụ lỗi không làm sập agent; 5 nhiệm vụ mẫu cho agent (`agents/tasks.py`).
- `local_ai/runtime`: runtime chạy việc gộp từ repo Agent: chạy lệnh dạng list không qua shell, `TerminalTool` cho agent (allowlist, tắt mặc định), hàng đợi job và gateway HTTP (tắt mặc định). Chi tiết gộp và rủi ro bảo mật: `docs/GOP_AGENT.md`.
- `local_ai/colab.py`: hàm `run` cho notebook Colab. Lệnh chạy không qua shell; lệnh lỗi thì ô báo đỏ và Run all dừng.
- `local_ai/check.py`: lệnh kiểm tra chung `python -m local_ai.check` (test, compileall, secret-scan).
- `local_ai/config`, `local_ai/experiments`, `local_ai/memory`: nạp danh sách model, ghi lại lượt chạy (`RunTracker`), bộ nhớ hội thoại ngắn.
- `configs/`: mọi file cấu hình (model, dataset, preset, huấn luyện). `data/`: dữ liệu mẫu (`data/raw/`) và bộ eval (`data/eval/`). `tests/`: test theo từng mốc. `docs/ARCHITECTURE.md`: kiến trúc. `docs/TRAIN_COLAB.md`: train trên Colab bằng iPhone.
- `notebooks/`: notebook Colab, sinh từ `notebooks/build.py` (lưu không kèm output).
- `archive/`: phần đã cất (corpus 10T token, hướng dẫn nanoGPT cũ, code gốc của repo Agent), kế hoạch tuần 1–2 (`tasks-tuan-1-2.md`) và nhật ký chi tiết tuần 1–3 (`memory-tuan-*.md`).

## Cài đặt và kiểm tra
Chạy mọi lệnh trong README từ thư mục gốc của repo, vì các đường dẫn trong file cấu hình (`dataset_path`, `output_dir`...) tính từ đó.

```bash
python -m pip install datasets                        # tải dataset Hugging Face
python -m pip install torch transformers trl peft     # huấn luyện (cần GPU)
# máy không có GPU: cài torch bản CPU trước, rồi mới cài các thư viện kia
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install bitsandbytes                    # nén 4bit/8bit và QLoRA (cần GPU CUDA)
python -m pip install tokenizers nbformat             # test model tí hon và test notebook
python -m local_ai.check                              # test + compileall + secret-scan
```

`python -m local_ai.check` chạy toàn bộ test (`unittest`), `compileall` thư mục `local_ai` và `secret-scan`, rồi in "KẾT QUẢ: XANH" hoặc "CHƯA XANH" (mã thoát khác 0).
- Test bị bỏ qua vì thiếu thư viện (torch, transformers, trl, peft, datasets, tokenizers, nbformat) cũng tính là chưa xanh, vì khi đó các test train thật trên CPU không chạy. Máy không có GPU thì cài torch bản CPU trước (dòng chú thích ở trên), rồi cài các thư viện kia và chạy lại.
- `--allow-skip` cho phép bỏ qua các test đó, chỉ để xem nhanh; không dùng để đánh dấu mốc Xong hay gộp PR.
- Từng lệnh riêng vẫn chạy được: `python -m unittest discover -s tests -v`, `python -m compileall -q local_ai`, `python -m local_ai.data secret-scan`.

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
Bảng in ra trọng số ở bf16/8bit/4bit và bộ nhớ khi train LoRA/QLoRA. Ví dụ model chính (27,78 tỷ tham số): trọng số bf16 khoảng 55,6 GB, 4bit khoảng 15,6 GB, train QLoRA khoảng 34,6 GB. Phần QLoRA đã hiệu chỉnh theo số đo thật của model 4B trên T4 (M15, xem bảng số đo thật); model 27B chưa đo, nên con số này chỉ là suy ra.

## Chạy model qua server local (Ollama, llama.cpp, vLLM)
Model có `backend: "openai_compatible"` được gọi qua `POST {base_url}/chat/completions` theo chuẩn OpenAI, chỉ dùng thư viện chuẩn của Python. Khi đó:
- `source` là tên model trên server;
- `timeout_s` là số giây chờ tối đa;
- `api_key_env` (tùy chọn) là **tên** biến môi trường chứa khóa, dùng khi server yêu cầu khóa. Không ghi khóa vào file cấu hình.

`base_url` chỉ được là máy này hoặc mạng nội bộ (localhost, 192.168.x.x, 10.x.x.x...). Địa chỉ Internet bị từ chối, để không vô tình gọi API trả phí.

`configs/models/platform.json` có sẵn 2 mục mẫu:
- `ollama`: `http://localhost:11434/v1`, model `huihui_ai/Qwen3.8-abliterated:27b`;
- `llamacpp`: `http://localhost:8080/v1`.

Cần cài Ollama, llama.cpp hoặc vLLM trước. Các lệnh `ollama`, `llama-server`, `vllm` dưới đây chưa chạy thử trong môi trường phát triển của repo, vì cần cài phần mềm và tải model; riêng Ollama với `qwen3:4b` đã chạy thật trên Colab T4 (26/9, notebook `agent_colab`). Phần của repo gọi tới server đã được test bằng server giả.

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

Kết quả đọc giấy phép và nguồn gốc của 3 dataset (ngày 25/9) nằm trong [`docs/GIAY_PHEP_DATASET.md`](docs/GIAY_PHEP_DATASET.md). Tóm tắt: `code` phải ghi nguồn; `vietnamese` được dịch từ dữ liệu gốc có giấy phép **cấm dùng thương mại**, nên chỉ dùng cá nhân hoặc nghiên cứu.

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

Với `--hub-repo <tên>/<repo> --hub-path <thư mục>`, chấm xong thì báo cáo được đẩy lên repo Hugging Face riêng tư. Lần chạy sau, nếu repo đã có báo cáo cùng cài đặt (model, `max_new_tokens`, mã băm bộ câu hỏi) thì báo cáo được dùng lại, không nạp model.

So sánh 2 lần chấm (ví dụ trước và sau khi train): `python -m local_ai.evaluation.compare <trước>/report.json <sau>/report.json`. Lệnh in bảng tỉ lệ đạt theo nhóm và theo ngôn ngữ, mức thay đổi (điểm %), cùng các câu mới đạt và mới trượt. `--max-new-tokens` ghi đè độ dài câu trả lời tối đa để chấm nhanh hơn; `--dry-run` chỉ in kế hoạch, không nạp model.

Nhóm câu eval (`code`, `reasoning`, `tool_use`) là cách chia riêng của bộ eval, khác với `domain` của dữ liệu train (`coding`, `math_logic`, `chat`...).

Nếu model lỗi (ví dụ server chưa chạy), lệnh dừng và ghi `status: "error"`. Model chạy bằng transformers mà máy thiếu torch hoặc transformers thì lệnh báo bỏ qua.

**Chặn trùng train/eval** theo id, theo nội dung, và theo câu hỏi đã chuẩn hóa (kể cả các lượt user trong hội thoại):
- Bước build dữ liệu (`hf-sft`, `build`) dừng và báo lỗi nếu dữ liệu train trùng với file trong `eval_sources`. Các cấu hình có sẵn đã chặn cả `seed_eval.jsonl` lẫn `eval_v1.jsonl`.
- Lệnh eval với `--train-data <train.jsonl hoặc sft.jsonl>` từ chối chạy nếu dữ liệu train chứa câu eval.

`PythonSandbox` chỉ chạy code trong một tiến trình riêng, có giới hạn thời gian. Nó không phải lớp cách ly an toàn. Nếu chấm code của model lạ, hãy chạy trong container.

## Train trên GPU: smoke → light → primary
Làm lần lượt từ model nhỏ đến model lớn; mỗi bước chạy `--dry-run` trước để xem cấu hình. Lệnh train đã chạy thật trên GPU T4 của Colab với `smoke` và `light` (QLoRA, `configs/training/colab_*.json`, 25–26/9) và trên CPU với model tí hon. Các cấu hình trong bảng dưới (LoRA bf16, model chính 27B) chưa chạy trên GPU.

Chuẩn bị:
- trên máy GPU, cài torch bản CUDA, rồi `python -m pip install transformers trl peft datasets bitsandbytes`;
- tạo dữ liệu vào đúng đường dẫn mà các cấu hình train đang đọc:

```bash
python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --output data/processed/hf_sft
```

| Bước | Model | Lệnh train | VRAM ước tính | GPU gợi ý |
| --- | --- | --- | ---: | --- |
| 1. smoke | `smoke` (Qwen2.5-0.5B), LoRA bf16 | `python -m local_ai.training.finetune --config configs/training/sft.json` | 2,2 GB | GPU bất kỳ từ 4 GB (T4 tự dùng fp16) |
| 2. light | `light` (Qwen3-4B), LoRA bf16 | `python -m local_ai.training.finetune --config configs/training/sft.json --base-model light --output-dir .runs/sft_light` | 11,0 GB | T4 16 GB, RTX 3060 12 GB; thêm `--quantization 4bit` (QLoRA) thì còn 9,2 GB |
| 3. primary | `primary` (27,78 tỷ), QLoRA 4bit | `python -m local_ai.training.finetune --config configs/training/qlora_primary.json` | 34,6 GB | 40–48 GB (A100 40 GB, L40S hoặc RTX A6000 48 GB); suy ra từ số đo model 4B, cần đo lại khi train thật trên GPU đó. LoRA bf16 không nén cần khoảng 70,5 GB (A100 80 GB) |

**Model chính là model ảnh + chữ (M17).** `primary` có kiến trúc Qwen3.5 (`Qwen3_5ForConditionalGeneration`): phần xử lý ảnh nằm ở `model.visual` (các khối vision và merger), phần ngôn ngữ ở `model.language_model`. Dữ liệu train chỉ có chữ, nên với model `kind: "multimodal"` lệnh train chỉ gắn LoRA vào phần ngôn ngữ: `target_modules: "all-linear"` kèm `exclude_modules` loại trừ phần xử lý ảnh (khớp tên `visual`, `vision_tower`, `multi_modal_projector`...). Trước M17, `all-linear` gắn LoRA vào cả phần xử lý ảnh. Kiểm tra trước khi train:
```bash
python -m local_ai.training.finetune --config configs/training/qlora_primary.json --dry-run   # mục "lora" ghi phạm vi gắn LoRA
```
Đã chạy thật trên CPU với model Qwen3.5 ảnh + chữ tí hon (`tests/test_m17_multimodal_lora.py`): LoRA có ở mọi lớp Linear của phần ngôn ngữ (attention thường, linear attention, MLP), không có ở phần xử lý ảnh hay `lm_head`. Model 27B thật chưa train (cần GPU 40–48 GB).

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
| `smoke` (Qwen2.5-0.5B) | `configs/training/colab_smoke.json`: QLoRA 4bit, fp16, batch 4, `max_length` 1024 | `smoke-colab` | khoảng 23 phút |
| `light` (Qwen3-4B) | `configs/training/colab_light.json`: QLoRA 4bit, fp16, batch 1 (tích lũy 16), `max_length` 2048, VRAM ước tính 9,2 GB (đo thật 8,2 GB) | `light-colab` | khoảng 121 phút |

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
10. đẩy adapter lên repo Hugging Face riêng tư;
11. in bảng số đo thật so với ước tính (thời gian train, VRAM, tốc độ chấm), rồi lưu vào `so_do/<model>.json` trong repo riêng tư.

**Cần chuẩn bị:** token Hugging Face quyền Write, lưu trong Colab Secrets (biểu tượng chìa khóa) với tên `HF_TOKEN`, bật Notebook access. Token không nằm trong notebook hay repo.

**Colab ngắt giữa chừng:** mở lại notebook, giữ nguyên model đã chọn và chạy Run all.
- Cờ `--push-to-hub` của lệnh train đẩy checkpoint mới nhất vào thư mục `last-checkpoint` của repo riêng tư sau mỗi `save_steps` bước (`light`: 10, `smoke`: 25).
- Lần chạy sau, lệnh train tự tải checkpoint đó về `.runs/colab_<model>/_hub/last-checkpoint` rồi train tiếp.
- Bước chấm trước khi train (`--hub-repo ... --hub-path eval/truoc`) cũng không phải chạy lại. Lần đầu chấm xong, báo cáo được đẩy lên repo riêng tư, kèm cài đặt (model, `max_new_tokens`, mã băm bộ câu hỏi). Lần sau, nếu cài đặt giống thì báo cáo được tải về dùng luôn.

**Lệnh lỗi thì dừng (M16).** Mọi lệnh trong notebook chạy qua `run(...)` của `local_ai/colab.py`, thay cho `!python`. Khi mã thoát khác 0, ô báo đỏ "Bước N lỗi (mã thoát ...)" và Run all dừng ngay, không chạy tiếp các ô sau như trước.

Notebook đẩy adapter lên repo riêng tư bằng lệnh con `push-adapter` của `local_ai.training.hub`. Token đọc từ biến môi trường `HF_TOKEN`; thêm `--dry-run` để chỉ kiểm tra tham số, không gọi mạng.

**Số đo thật (M15).** Ô cuối của notebook chạy `python -m local_ai.training.calibrate`, dùng các số đo sau:
- lệnh train ghi `measurements.json` trong thư mục đầu ra: GPU, VRAM đỉnh, thời gian, bước bắt đầu/kết thúc, số token của lần chạy;
- báo cáo eval ghi thời gian chấm và độ dài câu trả lời.

Lệnh in bảng "ước tính và đo thật", kèm hằng số đề xuất cho `estimate.py` và `vram.py`, nhưng không tự sửa code. Hệ số VRAM đo từ model nhỏ (`smoke`) không dùng được cho model lớn, vì ở model nhỏ phần bộ nhớ cho logits lớn hơn trọng số nhiều lần; lệnh sẽ ghi chú điều này.

**Số đo thật trên Colab** (GPU Tesla T4, file `so_do/<model>.json` trong repo riêng tư; bản chép của `smoke` ở `tests/fixtures/measurements/that_smoke_t4_2026-09-25.json`):

| Mục | `smoke` (25/9/2026) | `light` (25/9/2026) | Agent (`qwen3:4b`, 26/9/2026) |
| --- | --- | --- | --- |
| Train | 125 bước trong 14,9 phút (ước tính cũ 13,3); 5,36 TFLOPS hiệu dụng; 1.013.077 token | Colab ngắt sau bước 30, chạy tiếp được: 95 bước sau trong 66,7 phút (khoảng 42 giây/bước; ước tính cũ 56,9); 5,17 TFLOPS; 856.769 token. Lần chạy lại 26/9 không train thêm bước nào | — |
| VRAM khi train | 2,5 GB (ước tính cũ 1,3 GB, công thức mới 3,2 GB) | 8,2 GB (ước tính cũ 3,8 GB, công thức mới 9,2 GB) | — |
| Loss | 1,48 (bước 5) → 1,27 (bước 125) | 1,51 (bước 5) → 0,97 (bước 125) | — |
| Chấm 30 câu (tối đa 256 token/câu với `smoke`, 512 với `light`) | trước: 2,2 phút; sau: 2,6 phút (cả hai tính cả thời gian nạp model) | trước: 17,4 phút, tính cả nạp model (ước tính tối đa 17,2); sau: 2,5 phút (câu trả lời ngắn hơn hẳn, không còn phần suy nghĩ `<think>`) | — |
| Điểm eval trước → sau | 16/30 → 14/30 | 17/30 → 19/30; riêng tool_use 7/8 → 3/8 | — |
| Nhiệm vụ agent | — | — | 4/5 (80%) trong 11,1 phút; không đạt: `tong-cot-csv` |

Đã sửa theo số đo này:
- thông lượng train của T4 trong `estimate.py`: 6 → 5,2 TFLOPS;
- VRAM khi train QLoRA trong `vram.py` (chủ repo đồng ý ngày 26/9): công thức cũ ước tính `light` 3,8 GB, đo thật 8,2 GB. Phần thiếu là phần không tăng theo tỉ lệ số tham số: embedding bị đổi sang float32, và logits trên bộ từ vựng khoảng 152 nghìn token. Công thức mới cộng thêm 2,67 × √(tỷ tham số) GB cho model nén: `light` 9,2 GB, `smoke` 3,2 GB (hơi cao hơn số đo, an toàn khi chọn GPU). Với model chính 27B, ước tính tăng từ 20,5 lên 34,6 GB, nên GPU 24 GB có thể không đủ; con số này suy ra từ model 4B, cần đo lại khi làm M17.

Giữ nguyên (có số đo, không cần sửa): tốc độ chấm (`token_overhead_s`). Ước tính thời gian chấm tối đa của `light` (17,2 phút) đã khớp lần chấm trước khi train (17,4 phút, câu nào cũng sinh gần đủ 512 token). Lần chấm sau khi train có câu trả lời rất ngắn, nên số giây mỗi token (0,12) bị chi phí cố định của từng câu đẩy lên; không dùng số này để sửa. Từ nay lệnh eval nạp model trước khi bấm giờ và ghi riêng thời gian nạp (`load_s`).

Hệ số VRAM trong `so_do/light.json` (2,07) không dùng được: lần chạy lại 26/9 không train bước nào, nên VRAM đo được (4,68 GB) chỉ là lúc nạp model. Số đo lúc train thật là 8,17 GB (lần 25/9). Từ nay lệnh train không ghi đè số đo của lần đã train khi checkpoint đã đủ bước, và `calibrate` bỏ qua VRAM của lần chạy 0 bước.

**Vì sao tool_use tụt sau khi train, và cách giữ:**
- Nguyên nhân có thể: 2000 dòng train (code, toán, hội thoại tiếng Việt) không có dòng nào trả lời bằng JSON gọi công cụ, và gần như không có phần suy nghĩ `<think>` (chỉ preset `reasoning` có). Sau khi train, model trả lời ngắn, bỏ suy nghĩ, và quên dạng JSON `{"tool": ..., "arguments": ...}` mà bộ chấm yêu cầu. Chỉ 8 câu tool_use nên mỗi câu là 12,5%, nhưng tụt 4 câu là rõ.
- Cách 1 (nên làm trước): trộn khoảng 10% dữ liệu gọi công cụ vào dữ liệu train (khoảng 200/2000 dòng). Dữ liệu này sinh tự động từ mẫu có sẵn trong repo (calculator, read_file, search, và câu không cần công cụ trả `null`), không dùng API trả phí. Phải kiểm tra không trùng câu chấm bằng `--train-data` và MinHash của M12.
- Cách 2: giữ phần suy nghĩ: thêm dòng có `<think>` cho câu hỏi thường, hoặc khi chấm tắt chế độ suy nghĩ của Qwen3 cho cả trước và sau, để so sánh công bằng.
- Cách 3: train nhẹ tay hơn để model ít quên: learning rate 2e-4 → 1e-4, hoặc LoRA `r` 16 → 8.
- Cách 4: tăng số câu tool_use trong bộ chấm (đề xuất M19), để biết thay đổi nào là thật, không phải ngẫu nhiên.

Chủ repo đã chọn làm (26/9): cách 1 ở M20 (dữ liệu); cách 2 và 4 ở M19 (chấm công bằng hơn). Cách 3 chưa chọn.

Chấm `light` trước khi train: code chỉ đạt 1/8 câu, vì Qwen3-4B "suy nghĩ" trong khối `<think>` hết 512 token trước khi kịp viết code. Đây là giới hạn của bộ chấm, không phải lỗi train.

Điểm sau khi train của `smoke` giảm 2 câu (16 → 14 trên 30). Với 30 câu, chênh 2 câu có thể chỉ là ngẫu nhiên. Báo cáo chấm sau khi train không được đẩy lên Hugging Face, nên chưa xem được câu nào sai thêm.

**Ước tính thời gian** (`python -m local_ai.training.estimate`) không tải model, chỉ dựa vào cấu hình và `sft.jsonl`:
- số token mỗi dòng, kể cả phần đệm khi batch lớn hơn 1;
- thông lượng train của T4 (5,2 TFLOPS, đo khi train `smoke` và `light`).

Đây là ước lượng thô: thông lượng train và VRAM QLoRA đã sửa theo số đo thật (bảng trên); tốc độ chấm giữ nguyên vì đã khớp số đo; cách tính ghi ở đầu `local_ai/training/estimate.py`.

Nút Colab mở notebook ở nhánh `main`. Notebook đã chạy thật trên Colab T4 với cả `smoke` và `light` (25–26/9). Test `tests/test_m10_colab.py` và `tests/test_m11_colab_light.py` chỉ kiểm tra notebook hợp lệ và chạy các lệnh của nó bằng `--dry-run` với cả 2 model, ví dụ:

```bash
python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --total 2000 --output data/processed/hf_sft --dry-run
python -m local_ai.training.estimate --config configs/training/colab_light.json --max-new-tokens 512 --dry-run
python -m local_ai.training.finetune --config configs/training/colab_light.json --push-to-hub --hub-model-id ten-ban/huyen-light-qlora --dry-run
python -m local_ai.evaluation --model light-colab --max-new-tokens 512 --dry-run
python -m local_ai.evaluation.compare .runs/eval/light/truoc/report.json .runs/eval/light/sau/report.json --dry-run
python -m local_ai.training.calibrate --config configs/training/colab_light.json --max-new-tokens 512 --dry-run
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

**Đã chạy với Ollama thật trên Colab T4 (26/9/2026): 4/5 nhiệm vụ đạt (80%) trong 11,1 phút.** Nhiệm vụ không đạt là `tong-cot-csv`: model chạy `awk -F, 'NR > 1 {print $2}' du_lieu.csv`, TerminalTool từ chối đúng quy tắc vì lệnh có ký tự điều khiển shell (`>`). Sau đó model không thử lại bằng `cat` mà gọi calculator với một giá trị (30), nên trả lời 30 thay vì 87. Kết quả chép vào `tests/fixtures/measurements/that_agent_t4_2026-09-26.json`.

Test `tests/test_m13_agent_colab.py` (chạy không cần Ollama) gồm:
- kiểm tra notebook hợp lệ và chạy lệnh của nó bằng `--dry-run`;
- chạy 5 nhiệm vụ với công cụ thật, qua một server HTTP giả nói chuẩn OpenAI.

## Lộ trình tiếp theo (đề xuất, chưa làm)
Tuần 1 đề xuất 6 việc. Tuần 2 chuẩn bị chạy thật trên GPU và thử Ollama (notebook Colab train và agent); tuần 3 đã chạy thật trên Colab T4 (M15). Các việc còn lại gom vào 7 mốc đề xuất cho tuần 3 dưới đây. Đây chỉ là đề xuất, chủ repo chọn việc nào thì mới đưa vào `TASKS.md`:
1. **M15 – Số đo thật trên Colab** (xong 26/9: đủ số đo `smoke`, `light` và agent; bảng số đo ở mục "Train trên Colab miễn phí"). Dựa trên kết quả chủ repo chạy `train_colab` (smoke, light) và `agent_colab`: sửa hằng số ước tính trong `local_ai/training/estimate.py` và `local_ai/models/vram.py`, rồi ghi bảng số đo thật (thời gian, VRAM, điểm trước/sau, tỉ lệ agent) vào README.
2. **M16 – Notebook bền hơn** (xong):
   - dừng Run all khi một lệnh `!python` lỗi;
   - lưu báo cáo chấm trước lên repo Hugging Face, để chạy lại sau khi Colab ngắt không phải chấm lại.
3. **M17 – Model chính (ảnh + chữ)** (xong 26/9): chỉ gắn LoRA vào phần ngôn ngữ, không gắn vào phần xử lý ảnh. (Việc đổi `torch_dtype` sang `dtype` theo transformers 5.x đã làm sớm trong lượt rà soát M1–M16.) Train thật model 27B trên GPU 40–48 GB vẫn chờ chủ repo.
4. **M18 – Dùng model đã train trong agent:** gộp adapter vào model gốc, xuất GGUF để chạy bằng Ollama, rồi cho agent làm 5 nhiệm vụ mẫu với model vừa train, so với model gốc.
5. **M19 – Chấm công bằng hơn** (chủ repo chọn 26/9, làm trước; tiêu chí trong `TASKS.md`):
   - tùy chọn tắt chế độ suy nghĩ của Qwen3 khi chấm (`enable_thinking=False`), ghi vào cài đặt chấm để không dùng lại nhầm báo cáo cũ; chấm trước và sau dùng cùng cài đặt;
   - chấm sau khi train (Bước 9) đẩy báo cáo lên Hub ở `eval/sau`, luôn chấm lại (`--no-reuse`);
   - câu tool_use trong bộ chấm tăng từ 8 lên ít nhất 16;
   - TerminalTool từ chối lệnh thì gợi ý cách khác; thêm 1 nhiệm vụ agent phải thử lại sau khi bị từ chối (đo 26/9: model không thử lại bằng lệnh khác).
6. **M20 – Dữ liệu giữ khả năng gọi công cụ** (chủ repo chọn 26/9, làm sau M19; tiêu chí trong `TASKS.md`):
   - `hf-sft` trộn khoảng 10% dòng gọi công cụ tự sinh, đúng dạng JSON bộ chấm yêu cầu, có cả câu không cần công cụ, khác câu và số liệu của bộ chấm; notebook train bật mặc định (đo 26/9: tool_use của `light` tụt 7/8 → 3/8 sau khi train);
   - chặn gần trùng giữa dữ liệu train và bộ chấm bằng MinHash của M12, `manifest.json` ghi số dòng bị loại;
   - (chưa chọn) thêm một preset tiếng Việt nữa, sau khi đọc kỹ giấy phép.
7. **M21 – Tổng kết tuần 3.**

PR #4 (đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B): **đề nghị đóng**, vì xung đột khoảng 65 file và xóa phần agent; phần sửa dữ liệu hữu ích của nó đã có trong `main`. Chủ repo tự đóng.

Kế hoạch tuần 1–3 và bằng chứng từng mốc: tuần 3 trong `TASKS.md`, tuần 1–2 trong `archive/tasks-tuan-1-2.md`.
