# Kiến trúc

## Phạm vi
Repo dùng để **fine-tune và đánh giá model có sẵn trên Hugging Face**, chạy cục bộ. Model chính là `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (ảnh + chữ). Repo không chứa trọng số model và không gọi API trả phí.

Mọi bước đều điều khiển bằng file cấu hình JSON trong `configs/`. Các phần nằm ngoài repo được chọn qua cấu hình: trọng số model, dataset, server model và khóa truy cập (chỉ đọc từ biến môi trường).

Luồng chính:

```
preset / dataset Hugging Face ──► local_ai.data (kiểm tra, loại trùng, chặn trùng với eval) ──► sft.jsonl
sft.jsonl ──► local_ai.training.finetune (full | LoRA | QLoRA) ──► adapter (.runs/<tên>/adapter)
model gốc + adapter_path ──► local_ai.evaluation (30 câu, 4 cách chấm) ──► .runs/eval/<model>-<thời điểm>/report.{json,md}
```

## Thành phần

### Cấu hình (`configs/`, `local_ai/config`)

| File | Nội dung |
| --- | --- |
| `configs/models/platform.json` | Danh sách model: `backend` (`transformers` hoặc `openai_compatible`), `kind` (`text`/`multimodal`), `params_b`, `quantization`, `adapter_path`, `max_new_tokens`, `base_url`, `timeout_s`, `api_key_env`. |
| `configs/datasets/presets/*.json` | 3 preset dataset (`code`, `reasoning`, `vietnamese`): commit cố định, đọc kiểu streaming, `limit` nhỏ, giấy phép kèm trạng thái "cần kiểm tra lại". |
| `configs/datasets/hf_sft.json` | Mẫu để tự khai báo một dataset. |
| `configs/training/sft.json`, `qlora_primary.json` | Cấu hình huấn luyện: model gốc, `method`, `quantization`, siêu tham số, `max_steps`, `require_gpu`. |

`ModelConfig` kiểm tra giá trị ngay khi nạp; giá trị sai thì báo lỗi bằng tiếng Việt. Test `test_every_config_key_is_read_by_code` bảo đảm không có khóa cấu hình nào thừa.

### Dữ liệu (`local_ai/data`)
- `hub.py`:
  - đọc dataset Hugging Face; đọc kiểu streaming với `limit` để không tải cả dataset;
  - ánh xạ cột sang schema của repo. Hội thoại nhiều lượt được giữ nguyên, hiểu cả dạng `role/content` lẫn ShareGPT `from/value`;
  - trộn nhiều preset theo tỉ lệ.
- `core.py`:
  - kiểm tra schema và loại trùng theo nội dung;
  - chặn trùng với eval (`find_eval_overlap`) theo id, theo nội dung và theo câu hỏi đã chuẩn hóa;
  - xuất `sft.jsonl`: `reasoning` được đưa vào khối `<think>`, `context` đặt trước câu hỏi.
- `secrets.py`: quét khóa bí mật trong những file có thể bị commit.
- Lệnh: `python -m local_ai.data hf-sft | list-presets | build | secret-scan ...`.

### Model (`local_ai/models`)
- `adapters.py` — `HuggingFaceModelAdapter`:
  - chọn lớp nạp theo `kind`: `AutoModelForCausalLM` cho model chữ; `AutoProcessor` + `AutoModelForMultimodalLM` cho model ảnh + chữ (bản transformers cũ thì dùng `AutoModelForImageTextToText`);
  - nén 4bit/8bit bằng bitsandbytes;
  - GPU không có bf16 thì dùng fp16;
  - nạp LoRA đã train qua `adapter_path`;
  - thư viện nặng chỉ được import khi thật sự dùng tới.
- `openai_compatible.py` — `OpenAICompatibleAdapter`:
  - gọi `POST {base_url}/chat/completions` (Ollama, llama.cpp, vLLM) bằng urllib;
  - chỉ cho phép máy này hoặc mạng nội bộ; không dùng proxy, không đi theo redirect; có timeout và báo lỗi rõ.
- `router.py`:
  - `create_adapter` chọn adapter theo `backend`;
  - `ModelRouter` chọn model theo khả năng;
  - `ScriptedModelAdapter` là model giả dùng trong test và demo.
- `vram.py`: ước tính VRAM từ `params_b`, không tải model.

### Huấn luyện (`local_ai/training`)
`finetune.py` là khung SFT dùng transformers, trl và peft:
- chạy `full`, `lora`, hoặc QLoRA (`lora` cùng `quantization: "4bit"`);
- chạy tiếp từ `checkpoint-N` mới nhất;
- thiếu GPU hoặc thư viện thì trả `"status": "skipped"`;
- từ chối model chạy qua server và từ chối full fine-tune trên model đã nén.

`RunTracker` ghi cấu hình, chỉ số và đường dẫn adapter vào thư mục của lượt chạy.

### Đánh giá (`local_ai/evaluation`)
- `suite.py`:
  - 4 cách chấm: `exact`, `contains`, `regex`, `python_tests`. `python_tests` chạy unit test trong `PythonSandbox`, có giới hạn thời gian;
  - bỏ khối `<think>` trước khi chấm;
  - báo cáo theo nhóm và theo ngôn ngữ.
- Lệnh: `python -m local_ai.evaluation --model <tên> | --scripted`. Tùy chọn `--train-data` từ chối chạy nếu dữ liệu train chứa câu eval.
- Bộ đề `data/eval/eval_v1.jsonl`: 30 câu Việt + Anh, mỗi câu có đáp án mẫu.
- `benchmarks.py` là phần chấm khớp đúng cũ, vẫn giữ lại.

### Agent và công cụ (`local_ai/agents`, `local_ai/tools`)
`AutonomousAgent` là vòng lặp có giới hạn số lần: lập kế hoạch → chọn công cụ → thực thi → đánh giá → sửa hoặc thử lại. Agent dùng để thử model:
- công cụ lỗi thì agent ghi vào trace rồi thử lại;
- model lỗi (ví dụ server chưa chạy) thì agent dừng và trả kết quả chưa hoàn thành;
- `extract_json_object` tách được JSON dù câu trả lời có chữ thừa, khối code hoặc `<think>`.

`python -m local_ai.demo [--model <tên>]` là demo agent dùng công cụ máy tính.

## Kiểm thử
- `tests/test_m1_…` đến `tests/test_m7_…` tương ứng 7 mốc trong `TASKS.md`. Test không cần mạng hay GPU.
- `tests/test_m6_cpu_pipeline.py` chạy thật cả chuỗi (dữ liệu → LoRA → adapter → eval) trên CPU với model tí hon tự tạo; máy thiếu thư viện thì test tự bỏ qua.
- Lệnh kiểm tra trước khi đẩy: xem `CLAUDE.md`.

## An toàn và giới hạn
- Chỉ gọi model cục bộ hoặc server trong mạng nội bộ, không gọi API trả phí. Khóa chỉ đọc từ biến môi trường.
- `PythonSandbox` chỉ tách code ra một tiến trình riêng, có giới hạn thời gian. Nó không cách ly an toàn trước code độc hại; khi chấm code của model lạ, hãy chạy trong container.
- Chưa kiểm chứng trên GPU thật: QLoRA, model ảnh + chữ, và số VRAM ước tính. Chi tiết trong `memory.md`.
- Interface giao dịch (`TradingAnalysisTool`) chỉ để phân tích, không đặt lệnh.
