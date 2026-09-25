# Kiến trúc

## Phạm vi
Repo dùng để **fine-tune và đánh giá model có sẵn trên Hugging Face**, chạy cục bộ hoặc trên Colab miễn phí. Model chính là `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (ảnh + chữ). Repo cũng có một agent nhỏ dùng công cụ (calculator, `TerminalTool`) để thử model. Repo không chứa trọng số model và không gọi API trả phí.

Mọi bước đều điều khiển bằng file cấu hình JSON trong `configs/`. Các phần nằm ngoài repo được chọn qua cấu hình: trọng số model, dataset, server model và khóa truy cập (chỉ đọc từ biến môi trường).

Luồng chính:

```
preset / dataset Hugging Face ──► local_ai.data (kiểm tra, loại trùng, lọc chất lượng, chặn trùng với eval) ──► sft.jsonl
sft.jsonl ──► local_ai.training.finetune (full | LoRA | QLoRA) ──► adapter (.runs/<tên>/adapter)
model gốc + adapter_path ──► local_ai.evaluation (30 câu, 4 cách chấm) ──► .runs/eval/<model>-<thời điểm>/report.{json,md}
```

Trên Colab (tuần 2), `notebooks/train_colab.ipynb` gọi đúng các lệnh trên. Trong lúc train, checkpoint được đẩy lên repo Hugging Face riêng tư để chạy tiếp khi Colab ngắt. `notebooks/agent_colab.ipynb` chạy Ollama ở `localhost` rồi cho agent làm các nhiệm vụ mẫu:

```
Ollama (qwen3:4b, localhost:11434) ◄── adapter kiểu OpenAI ◄── AutonomousAgent ──► calculator, TerminalTool (thư mục làm việc riêng)
```

## Thành phần

### Cấu hình (`configs/`, `local_ai/config`)

| File | Nội dung |
| --- | --- |
| `configs/models/platform.json` | Danh sách model: `backend` (`transformers` hoặc `openai_compatible`), `kind` (`text`/`multimodal`), `params_b`, `quantization`, `adapter_path`, `max_new_tokens`, `base_url`, `timeout_s`, `api_key_env`. |
| `configs/datasets/presets/*.json` | 3 preset dataset (`code`, `reasoning`, `vietnamese`): commit cố định, đọc kiểu streaming, `limit` nhỏ, giấy phép kèm trạng thái "cần kiểm tra lại". |
| `configs/datasets/hf_sft.json` | Mẫu để tự khai báo một dataset. |
| `configs/datasets/quality.json` | Ngưỡng của 4 bộ lọc chất lượng dữ liệu (ngôn ngữ, độ dài, lặp, gần trùng). |
| `configs/training/sft.json`, `qlora_primary.json`, `colab_smoke.json`, `colab_light.json` | Cấu hình huấn luyện: model gốc, `method`, `quantization`, `dtype`, siêu tham số, `max_steps`, `require_gpu`, đẩy checkpoint lên Hub (`push_to_hub`, `hub_model_id`, `hub_private`). |

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
- `quality.py`: bộ lọc chất lượng, cấu hình ở `configs/datasets/quality.json`. `hf-sft` bật mặc định, `build` bật bằng `--quality`.
  - Các bộ lọc: ngôn ngữ (ưu tiên tiếng Việt: nhận theo chữ có dấu, loại tiếng Việt không dấu và chữ không phải Latin), độ dài, lặp từ/câu (xét từng lượt, bỏ qua code và lệnh LaTeX), gần trùng (MinHash tự viết kiểu một hoán vị, LSH theo dải, so lại bằng Jaccard thật).
  - Chạy sau bước loại trùng tuyệt đối, trước bước chặn trùng với eval. Thống kê trước/sau lọc ghi vào mục `quality` của `manifest.json`.
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
- `--push-to-hub --hub-model-id <tên>/<repo>`: Trainer đẩy checkpoint mới nhất vào thư mục `last-checkpoint` của repo riêng tư (`hub_strategy` checkpoint). Máy mới (ví dụ Colab vừa bị ngắt) không có `checkpoint-N` thì tải `last-checkpoint` về `output_dir/_hub/` rồi train tiếp. Thư mục bắt đầu bằng `_` nên Trainer không đẩy ngược nó lên Hub;
- thiếu GPU hoặc thư viện thì trả `"status": "skipped"`;
- từ chối model chạy qua server và từ chối full fine-tune trên model đã nén.

`hub.py`: kiểm tra tên repo, tải `last-checkpoint`, đọc `trainer_state.json` trên repo để biết đã train tới bước nào, đẩy adapter lên repo riêng tư (`python -m local_ai.training.hub push-adapter`). Token chỉ đọc từ biến môi trường `HF_TOKEN`; `huggingface_hub` chỉ được import khi gọi Hub.

`estimate.py` (`python -m local_ai.training.estimate`) ước tính thời gian train và chấm trên T4, không tải model:
- số token mỗi dòng đếm từ `sft.jsonl` (kể cả phần đệm khi batch lớn hơn 1);
- phép tính ≈ 2 × số tham số × số token × số lượt chạy qua model, chia cho thông lượng giả định của GPU;
- báo số bước đã train nếu có checkpoint trên máy hoặc trên Hub.

Các hằng số ghi ở đầu file và chưa đo trên T4 thật.

Số đo thật (M15):
- lệnh train ghi `measurements.json` trong `output_dir` (GPU, VRAM đỉnh, thời gian, bước bắt đầu/kết thúc, số token của lần chạy), trước khi lưu adapter, nên được đẩy lên Hub cùng adapter;
- báo cáo eval ghi `duration_s` và `output_chars`; báo cáo agent ghi `duration_s`;
- `calibrate.py` (`python -m local_ai.training.calibrate`) đặt số đo cạnh ước tính và tính ngược hằng số đề xuất (`train_tflops`, `TRAINING_FACTOR`, `token_overhead_s`), không tự sửa code;
- hệ số VRAM đo từ model nhỏ được đánh dấu là không dùng được.

`RunTracker` ghi cấu hình, chỉ số và đường dẫn adapter vào thư mục của lượt chạy.

### Đánh giá (`local_ai/evaluation`)
- `suite.py`:
  - 4 cách chấm: `exact`, `contains`, `regex`, `python_tests`. `python_tests` chạy unit test trong `PythonSandbox`, có giới hạn thời gian;
  - bỏ khối `<think>` trước khi chấm;
  - báo cáo theo nhóm và theo ngôn ngữ.
- Lệnh: `python -m local_ai.evaluation --model <tên> | --scripted`. Tùy chọn `--train-data` từ chối chạy nếu dữ liệu train chứa câu eval.
- `--max-new-tokens` ghi đè độ dài câu trả lời; `--dry-run` chỉ in kế hoạch, không nạp model.
- `compare.py` (`python -m local_ai.evaluation.compare TRUOC SAU`): bảng so sánh 2 báo cáo theo nhóm và ngôn ngữ, kèm câu mới đạt và câu mới trượt.
- Bộ đề `data/eval/eval_v1.jsonl`: 30 câu Việt + Anh, mỗi câu có đáp án mẫu.
- `benchmarks.py` là phần chấm khớp đúng cũ, vẫn giữ lại.

### Runtime (`local_ai/runtime`, gộp từ repo Agent)
Viết lại bằng thư viện chuẩn từ repo Agent (commit `78a3e25`; xem `docs/GOP_AGENT.md`):
- `executor.run_command`: chạy lệnh dạng list, **không qua shell**, trong thư mục làm việc; có timeout (quá giờ thì dừng cả nhóm tiến trình con); cắt output; ghi job id và thời điểm;
- `jobs.JobQueue`: hàng đợi job trong bộ nhớ (tạo, xem, nhận, trả kết quả), có giới hạn số job;
- `auth.require_token`: kiểm tra `Bearer <token>` bằng so sánh an toàn.

- `terminal.TerminalTool`: công cụ chạy lệnh cho agent.
  - Tắt mặc định; allowlist nằm trong `configs/tools/terminal.json`.
  - Chặn ký tự điều khiển shell, tham số nguy hiểm và đường dẫn ra ngoài thư mục làm việc; ghi log JSONL từng lệnh.
  - Đăng ký vào `ToolRegistry` bằng `register_terminal`.
- `gateway`: HTTP hàng đợi job bằng `http.server`. Tắt mặc định, chỉ nghe `127.0.0.1`, bắt buộc token; không chạy lệnh qua mạng.

Import không mở cổng mạng, không tạo thư mục.

### Notebook Colab (`notebooks/`)
`notebooks/build.py` sinh các notebook (không kèm output, thư viện ghim phiên bản, mỗi ô có chú thích tiếng Việt). Notebook chỉ gọi các lệnh `python -m local_ai...` của repo, nên test chạy được đúng các lệnh đó bằng `--dry-run`:
- `train_colab.ipynb`, trên GPU T4:
  1. chọn model `smoke` (Qwen2.5-0.5B) hoặc `light` (Qwen3-4B);
  2. lấy dữ liệu 2000 dòng;
  3. ước tính thời gian;
  4. chấm trước;
  5. QLoRA fp16 (`configs/training/colab_<model>.json`), đẩy checkpoint lên Hub; chạy lại thì train tiếp;
  6. chấm sau (`<model>-colab`);
  7. in bảng so sánh;
  8. đẩy adapter.

  Hướng dẫn trên iPhone: `docs/TRAIN_COLAB.md`.
- `agent_colab.ipynb`: cài Ollama (ghim bản 0.34.4) → tải `qwen3:4b` → chạy máy chủ ở `localhost:11434` → agent làm 5 nhiệm vụ mẫu qua adapter kiểu OpenAI (mục `ollama-colab`).

### Agent và công cụ (`local_ai/agents`, `local_ai/tools`)
`AutonomousAgent` là vòng lặp có giới hạn số lần: lập kế hoạch → chọn công cụ → thực thi → đánh giá → sửa hoặc thử lại. Agent dùng để thử model:
- công cụ lỗi thì agent ghi vào trace rồi thử lại;
- model lỗi (ví dụ server chưa chạy) thì agent dừng và trả kết quả chưa hoàn thành;
- `extract_json_object` tách được JSON dù câu trả lời có chữ thừa, khối code hoặc `<think>`.

`python -m local_ai.demo [--model <tên>]` là demo agent dùng công cụ máy tính.

Mỗi công cụ đăng ký kèm mô tả tiếng Anh: làm gì, nhận tham số nào (`ToolRegistry.register(name, tool, description)`). Agent gửi danh sách này, lịch sử quan sát và mẫu JSON cần trả về trong prompt, để model thật biết gọi công cụ thế nào.

`agents/tasks.py` (`python -m local_ai.agents.tasks`) chạy 5 nhiệm vụ mẫu trong `data/eval/agent_tasks_v1.jsonl` với `calculator` và `TerminalTool`:
- `TerminalTool` chỉ bật trong thư mục làm việc riêng của từng nhiệm vụ;
- nhiệm vụ đạt khi câu trả lời đúng và agent đã gọi mọi công cụ cần dùng;
- lệnh in trace và tỉ lệ thành công;
- model là server kiểu OpenAI (`--model`, ví dụ `ollama-colab`) hoặc câu trả lời mẫu (`--scripted`).

## Kiểm thử
- `tests/test_m1_…` đến `tests/test_m7_…` tương ứng 7 mốc tuần 1; `tests/test_m8_…` đến `tests/test_m14_…` là các mốc tuần 2 trong `TASKS.md`. Test không cần mạng hay GPU.
- `tests/test_m6_cpu_pipeline.py` chạy thật cả chuỗi (dữ liệu → LoRA → adapter → eval) trên CPU với model tí hon tự tạo; máy thiếu thư viện thì test tự bỏ qua.
- `tests/test_m8_runtime.py`, `tests/test_m9_terminal.py`:
  - kiểm tra runtime gộp từ repo Agent: chuỗi chèn lệnh chỉ được in ra như chữ, không được chạy;
  - `TerminalTool` chặn tham số nguy hiểm và đường dẫn ra ngoài thư mục làm việc;
  - gateway tắt mặc định;
  - kiểm tra bằng AST rằng không nơi nào gọi với `shell=True`.
- `tests/test_m10_colab.py`, `tests/test_m11_colab_light.py`:
  - notebook hợp lệ (nbformat) và khớp với `notebooks/build.py`;
  - mọi lệnh của notebook chạy được bằng `--dry-run`, với cả model `smoke` lẫn `light`;
  - Hugging Face Hub và thư viện train là module giả;
  - số phút ghi trong tài liệu khớp với ước tính.
- `tests/test_m12_quality.py` kiểm tra từng bộ lọc chất lượng bằng fixture trong `tests/fixtures/quality/` (mỗi dòng ghi kết quả mong đợi), cùng thống kê trước/sau lọc trong `manifest.json`.
- `tests/test_m13_agent_colab.py`:
  - kiểm tra notebook agent hợp lệ;
  - chạy 5 nhiệm vụ với công cụ thật, qua câu trả lời mẫu và qua server HTTP giả nói chuẩn OpenAI;
  - kiểm tra prompt có gửi danh sách công cụ.
- `tests/test_m14_summary.py` giữ README khớp code:
  - mọi lệnh và notebook đều có trong README;
  - nút Colab trỏ đúng file;
  - README nhắc tới mọi cấu hình train và tài liệu;
  - bảng tiến độ tuần 2 đầy đủ.
- `tests/test_consistency.py` giữ repo thống nhất: file mẫu dataset và preset cùng quy tắc (streaming, `limit` ≤ 1000, trạng thái giấy phép), cùng một thư mục `data/processed/hf_sft`, mọi lệnh con có trợ giúp, chữ cho người dùng bằng tiếng Việt.
- Lệnh kiểm tra trước khi đẩy: xem `CLAUDE.md`.

## An toàn và giới hạn
- Chỉ gọi model cục bộ hoặc server trong mạng nội bộ, không gọi API trả phí. Khóa chỉ đọc từ biến môi trường.
- `PythonSandbox` chỉ tách code ra một tiến trình riêng, có giới hạn thời gian. Nó không cách ly an toàn trước code độc hại; khi chấm code của model lạ, hãy chạy trong container.
- `TerminalTool` và gateway tắt mặc định. Lệnh `local_ai.agents.tasks` chỉ bật `TerminalTool` trong thư mục làm việc riêng của từng nhiệm vụ.
- Chưa kiểm chứng trên GPU hay model thật:
  - QLoRA và model ảnh + chữ;
  - số VRAM và thời gian ước tính;
  - 2 notebook Colab;
  - agent với Ollama + `qwen3:4b`.

  Chi tiết trong `memory.md`; lộ trình tuần 3 (đề xuất) ở cuối `README.md`.
- Interface giao dịch (`TradingAnalysisTool`) chỉ để phân tích, không đặt lệnh.
