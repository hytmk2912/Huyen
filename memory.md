# Bộ nhớ làm việc

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng), chỉ ghi những gì lượt sau cần biết. Kế hoạch và tiêu chí nằm trong `TASKS.md`.

## Checklist
- [x] M1 Sửa lỗi + dọn repo (xong 24/9)
- [x] M2 Nạp model đúng loại + nén 4-bit (xong 24/9)
- [x] M3 Adapter server local kiểu OpenAI (xong 24/9)
- [x] M4 Preset dataset (xong 24/9)
- [x] M5 Đánh giá (eval) (xong 24/9)
- [x] M6 Chạy thật trên CPU (xong 24/9)
- [ ] M7 Tổng kết

## Mốc đang làm
- Không có mốc dở. Mốc kế tiếp: **M7** (tổng kết: README, docs/ARCHITECTURE.md, lộ trình, % tiến độ, lệnh train khi có GPU smoke → light → primary kèm VRAM).

## Việc dở
- **PR #4** (nhánh `claude/expand-model-training-repo-v2yzyr`) chưa gộp và **đi ngược một phần kế hoạch này** (xoá hẳn agent và corpus thay vì sửa agent và cất corpus vào `archive/`). Phần còn dùng lại được: sửa 3 lỗi dữ liệu (commit `2626c7d`, cho M4, M5) và `local_ai/evaluation/suites.py` (cho M5). Chủ repo cần quyết định đóng hay gộp PR #4.
- **PR #5** (nhánh `claude/nhiem-vu-1-tuan`) chỉ có commit thiết lập, đã nằm trong PR #6; gộp PR #6 thì có thể đóng PR #5.
- Nhánh làm việc là `claude/nhiem-vu-tuan` (PR #6). Phiên chạy tự động được gán nhánh `claude/nhiem-vu-tuan-<mã>`, nhưng lượt vẫn đẩy lên `claude/nhiem-vu-tuan` theo yêu cầu nhiệm vụ.
- Yêu cầu đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B (bf16) mới có ở PR #4; `main` vẫn dùng Qwen2.5-Coder-7B, Qwen3-4B, Qwen2.5-0.5B.
- Adapter server local (M3) chưa gửi được ảnh: có ảnh thì báo lỗi rõ. Muốn gửi ảnh tới Ollama/vLLM cần thêm `image_url` dạng base64 (ngoài `TASKS.md`, hỏi chủ repo trước).
- Việc chủ repo tự làm (AI không làm được): đổi mật khẩu máy chủ cũ còn trong lịch sử commit; cân nhắc chuyển repo sang Private; xoá nhánh `codex/build-autonomous-ai-system-architecture` (đã gộp hết vào `main`).

## Lỗi gặp (chưa sửa)

| Lỗi | Nơi | Mốc sửa |
| --- | --- | --- |
| Chưa chạy với server thật: adapter M3 mới test bằng server giả; `ollama` và `llamacpp` trong `platform.json` chưa thử với Ollama/llama.cpp thật. | `local_ai/models/openai_compatible.py` | Khi có máy chạy server |
| Chưa chạy thật trên GPU: nạp model multimodal, nén 4bit và QLoRA mới chỉ test bằng module giả (M6 chỉ chạy thật LoRA model text trên CPU). `target_modules: "all-linear"` có thể gắn LoRA cả vào phần xử lý ảnh của model chính; cần xem lại khi có GPU. | `local_ai/models/adapters.py`, `local_ai/training/finetune.py` | Khi có GPU |
| transformers 5.17 cảnh báo `torch_dtype` đã cũ, nên đổi sang `dtype`. Vẫn chạy đúng; đổi khi không cần hỗ trợ bản transformers < 4.56. | `local_ai/models/adapters.py`, `local_ai/training/finetune.py` | Sau tuần 1 |

## Kết quả đọc thật preset (24/9, M4)
Cài `datasets` 5.0.1 vào venv riêng trong scratchpad (không cài vào môi trường test), mạng tới huggingface.co thông.
- `hf-sft --config presets/<tên>.json --limit 20`: `code` 20/20, `reasoning` 20/20, `vietnamese` 20/20 dòng hợp lệ, 0 bị loại. Hội thoại tiếng Việt có 6–12 lượt, giữ đủ.
- `hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --total 50`: đọc 20/15/15 dòng, 50 dòng vào `sft.jsonl`, khoảng 12 giây.
- Lỗi gặp: đọc stream dừng giữa chừng (`open-r1/OpenR1-Math-220k`, parquet lớn) làm Python crash lúc thoát ("Fatal Python error: PyGILState_Release", exit 134), dù file đã ghi xong. Lỗi nằm trong thư viện `datasets`, code `datasets` thuần cũng bị; chờ 15 giây trước khi thoát thì hết lỗi. Cách xử lý: `hf-sft` gọi `os._exit(0)` sau khi ghi xong, chỉ khi đã đọc stream thật (`real_stream_used`). Sau khi sửa, cả 3 preset thoát với mã 0.

## Kết quả chạy thật trên CPU (24/9, M6)
Máy: 4 CPU, 15 GB RAM, không GPU. Đã cài vào Python chính: torch 2.14.0+cpu (từ download.pytorch.org/whl/cpu, 28 giây), transformers 5.17.0, trl 1.13.0, peft 0.21.0, datasets 5.0.1, accelerate 1.15.0 (35 giây).
- `tests/test_m6_cpu_pipeline.py`: 8 dòng `sft.jsonl` → LoRA r=4, 2 bước, batch 2 → adapter → nạp lại → eval 30 câu (`max_new_tokens` 8). train_loss 5,993 (xấp xỉ ln 400, đúng với model ngẫu nhiên vocab 400), eval 0/30, tổng khoảng 6 giây. Toàn bộ 108 test chạy trong khoảng 9 giây.
- Chặn mạng (proxy sai) vẫn xanh, nên không tải gì. Venv trống (không thư viện): `OK (skipped=1)`.
- Chạy bằng CLI: `finetune --config` (2 bước, loss 5,993) rồi `evaluation --model tiny-lora --train-data sft.jsonl` (0/30, exit 0). Trainer in log từng bước ra stdout trước khối JSON kết quả.

## Nhật ký các lượt

| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 24/9 | Thiết lập | Tạo `TASKS.md`, `memory.md`, skill `lam-moc`; thêm quy tắc vào `CLAUDE.md`. Gốc trên `main`: 28 test qua (1 bỏ qua). |
| 24/9 | M1 | Xong 9/9. Công cụ lỗi không làm sập agent; `extract_json_object` tách JSON (chữ thừa, khối code, `<think>`); cất corpus 10T vào `archive/corpus-10t/`; tách `secret-scan` sang `secrets.py`; sửa link cũ; viết lại README. 34 test qua. |
| 24/9 | M2 | Xong 8/8. `platform.json` thêm `kind`, `params_b` (số liệu lấy từ Hugging Face: model chính `AutoModelForMultimodalLM`, 27,78 tỷ). Adapter: multimodal → `AutoProcessor` + `AutoModelForMultimodalLM` (dự phòng `AutoModelForImageTextToText`), gửi được ảnh; nén `4bit`/`8bit` qua bitsandbytes; GPU không bf16 tự chuyển fp16. QLoRA = `method: "lora"` + `quantization: "4bit"` (giữ nguyên test cũ yêu cầu từ chối `method: "qlora"`), mẫu `configs/training/qlora_primary.json`. Lệnh `python -m local_ai.models.vram`. Cập nhật skill `lam-moc` theo quy trình mới, thêm "Đọc memory.md" vào `CLAUDE.md`. Kiểm tra: 54 test qua, compileall và secret-scan sạch. Lỗi gặp: không có. Tiếp theo: M3. |
| 24/9 | M3 | Xong 7/7. `OpenAICompatibleAdapter` (urllib) gọi `/v1/chat/completions`. Chỉ cho địa chỉ local/mạng nội bộ, không dùng proxy, không theo redirect; có timeout; báo lỗi tiếng Việt khi server chưa chạy, hết thời gian chờ, HTTP lỗi hay trả sai chuẩn. `platform.json` thêm `backend`, `base_url`, `timeout_s`, `api_key_env`, 2 mục mẫu `ollama` (`huihui_ai/Qwen3.8-abliterated:27b`, đã kiểm tra tag trên ollama.com) và `llamacpp`. Agent dừng êm khi model lỗi; `create_adapter` + `ModelRouter.from_configs`; demo chạy qua adapter bằng `--model`. Fine-tune từ chối model server. README hướng dẫn Ollama/llama.cpp/vLLM. Lỗi gặp: test gọi sai hàm helper (trùng tham số `base_url`), đã sửa; server giả tắt chậm 0,5 s mỗi lần, đã giảm `poll_interval`. Kiểm tra: 71 test qua, compileall và secret-scan sạch. Tiếp theo: M4. |
| 24/9 | M4 | Xong 9/9. 3 preset `code`/`reasoning`/`vietnamese` (commit cố định, streaming, limit 1000, giấy phép "cần kiểm tra lại trên dataset card"). Sửa 2 lỗi dữ liệu: giữ nguyên hội thoại nhiều lượt kể cả system (hiểu cả ShareGPT from/value), giữ reasoning (`<think>`) và context trong `sft.jsonl`. `hf-sft --preset tên:tỉ_lệ` trộn nhiều preset vào một `sft.jsonl`; lệnh `list-presets`; thêm domain `chat`. Đọc thật 20 dòng/preset thành công (xem mục trên). Lỗi gặp: crash lúc thoát khi đọc stream, đã xử lý. Kiểm tra: 85 test qua, compileall và secret-scan sạch. Tiếp theo: M5. |
| 24/9 | M5 | Xong 8/8. `local_ai/evaluation/suite.py`: chấm `exact`/`contains`/`regex`/`python_tests` (PythonSandbox, timeout), bỏ `<think>` trước khi chấm. `data/eval/eval_v1.jsonl`: 30 câu (15 vi, 15 en; code 8, reasoning 14, tool_use 8), mỗi câu có `reference` và đều đạt. Lệnh `python -m local_ai.evaluation --model <tên>` hoặc `--scripted`, báo cáo JSON + Markdown vào `.runs/eval/`. Chặn trùng train/eval theo id, nội dung và câu hỏi chuẩn hóa (`find_eval_overlap`), ở bước build và qua `--train-data`. Lỗi gặp: test cũ `test_build_versions_and_prevents_eval_leakage` dựa vào lỗi cũ (train và eval cùng nội dung, khác id); đổi nội dung bản ghi eval trong test, giữ nguyên 2 kiểm tra. 50 dòng thật tải ở M4 không trùng eval. Kiểm tra: 105 test qua, compileall và secret-scan sạch. Tiếp theo: M6. |
| 24/9 | M6 | Xong 7/7. Cài torch CPU + transformers/trl/peft/datasets. Test `test_m6_cpu_pipeline.py` tự tạo tokenizer BPE + Llama 2 lớp, chạy thật fixture → sft.jsonl → LoRA 2 bước (CPU) → adapter → nạp lại → eval, khoảng 6 giây (xem mục trên). Thêm `max_steps` (finetune), `adapter_path` và `max_new_tokens` (ModelConfig); `train()` trả `steps`. Lỗi gặp: không có lỗi chặn; ghi lại cảnh báo `torch_dtype`. Kiểm tra: 108 test qua, compileall và secret-scan sạch. Tiếp theo: M7. |
