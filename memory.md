# Bộ nhớ làm việc

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng). Kế hoạch và tiêu chí: `TASKS.md`.

## XONG – đã gộp vào main
Cả 7 mốc đã xong ngày 24/9. PR #6 đã gộp vào `main` (commit gộp `63bdc61`) theo yêu cầu của chủ repo; PR #5 đã đóng vì thừa. Lượt tự động sau: không làm mốc mới, dừng ngay. Việc mới phải hỏi chủ repo trước.
Sau khi gộp, đã rà soát cả repo cho thống nhất: thư mục dữ liệu SFT, file mẫu dataset, trợ giúp lệnh, tài liệu (xem nhật ký).

## Checklist
- [x] M1 Sửa lỗi + dọn repo · [x] M2 Nạp model đúng loại + nén 4-bit · [x] M3 Adapter server local kiểu OpenAI
- [x] M4 Preset dataset · [x] M5 Đánh giá (eval) · [x] M6 Chạy thật trên CPU · [x] M7 Tổng kết

## Tiến độ từng phần
| Phần | Code + test | Đã chạy thật | Còn thiếu |
| --- | --- | --- | --- |
| Dữ liệu (preset, trộn, chặn trùng eval) | 100% | 100%: trộn 2500/2500 dòng thật, khoảng 15 giây | Đọc lại giấy phép 3 dataset |
| Fine-tune LoRA (model chữ) | 100% | 100% trên CPU (model tí hon), 0% trên GPU | Chạy smoke/light trên GPU |
| QLoRA, model ảnh + chữ | 100% (module giả) | 0% | Cần GPU 24 GB |
| Eval (30 câu, 4 cách chấm) | 100% | 100% với model tí hon | Chấm model thật |
| Server local (Ollama, llama.cpp, vLLM) | 100% (server giả) | 0% | Cần máy chạy server |
| Tài liệu (README, ARCHITECTURE) | 100% | 25 lệnh README đã kiểm cờ; lệnh repo đã chạy thử | — |
| **Tổng theo tiêu chí `TASKS.md`** | **100%** (7/7 mốc, 114 test) | | |

## Lệnh train khi có GPU (chưa chạy thử; VRAM là ước lượng của `python -m local_ai.models.vram`)
Chuẩn bị: cài torch bản CUDA, rồi `pip install transformers trl peft datasets bitsandbytes`. Tạo dữ liệu:
`python -m local_ai.data hf-sft --preset code:0.4 --preset reasoning:0.3 --preset vietnamese:0.3 --output data/processed/hf_sft`
1. smoke (Qwen2.5-0.5B, LoRA bf16, **khoảng 2,2 GB**): `python -m local_ai.training.finetune --config configs/training/sft.json`, rồi `python -m local_ai.evaluation --model smoke-lora --train-data data/processed/hf_sft/sft.jsonl`.
2. light (Qwen3-4B, LoRA bf16, **khoảng 11,0 GB**; thêm `--quantization 4bit` thì **khoảng 3,8 GB**): `python -m local_ai.training.finetune --config configs/training/sft.json --base-model light --output-dir .runs/sft_light`, rồi eval `--model light-lora`.
3. primary (27,78 tỷ, QLoRA 4bit, **khoảng 20,5 GB**, GPU 24 GB; LoRA bf16 không nén cần khoảng 70,5 GB): `python -m local_ai.training.finetune --config configs/training/qlora_primary.json`, rồi eval `--model primary-qlora`.
Chạy `--dry-run` trước mỗi bước. Đo VRAM thật, rồi sửa hệ số trong `local_ai/models/vram.py`.

## Việc dở (chủ repo quyết định)
- PR #4 (`claude/expand-model-training-repo-v2yzyr`) đi ngược kế hoạch: xoá hẳn agent và corpus. Hỏi chủ repo nên đóng hay lấy phần nào; đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B mới chỉ có ở PR #4.
- Việc ngoài `TASKS.md`, cần hỏi trước: gửi ảnh qua server local; lộ trình đề xuất trong README (6 việc).
- Việc chủ repo tự làm: đổi mật khẩu máy chủ cũ còn trong lịch sử commit; cân nhắc chuyển repo sang Private; xoá nhánh `codex/build-autonomous-ai-system-architecture`; đọc lại giấy phép 3 dataset preset.

## Lỗi còn tồn
| Lỗi | Nơi |
| --- | --- |
| Chưa chạy thật trên GPU: QLoRA, model ảnh + chữ. `target_modules: "all-linear"` có thể gắn LoRA cả vào phần xử lý ảnh của model chính. | `local_ai/models/adapters.py`, `local_ai/training/finetune.py` |
| Adapter server local chưa thử với Ollama hay llama.cpp thật; chưa gửi được ảnh. | `local_ai/models/openai_compatible.py` |
| transformers 5.17 cảnh báo `torch_dtype` đã cũ, nên đổi sang `dtype`. Code vẫn chạy đúng. | `adapters.py`, `finetune.py` |
| `datasets` 5.0.1 crash lúc thoát khi dừng đọc stream giữa chừng. Đã tránh bằng `os._exit(0)` sau khi ghi xong (chỉ khi đã đọc stream thật). | `local_ai/data/__main__.py` |

## Kết quả chạy thật (tóm tắt)
- M4, M7 (mạng tới huggingface.co thông):
  - `hf-sft --config presets/code.json`: 1000/1000 dòng, 5 giây;
  - trộn 0,4/0,3/0,3: đọc 1000/750/750 dòng, giữ 2500/2500 sau khi sửa lỗi id `"NaN"` ở preset `reasoning`, 15 giây;
  - không trùng câu eval nào.
- M6 (4 CPU, 15 GB RAM): torch 2.14.0+cpu, transformers 5.17.0, trl 1.13.0, peft 0.21.0, datasets 5.0.1.
  - Chuỗi fixture → LoRA 2 bước → adapter → nạp lại → eval: khoảng 6 giây, train_loss 5,993, eval 0/30 (model ngẫu nhiên).
  - Chặn mạng vẫn xanh. Venv trống: test M6 tự bỏ qua.

## Nhật ký các lượt (24/9)
| Lượt | Kết quả |
| --- | --- |
| Thiết lập | `TASKS.md`, `memory.md`, skill `lam-moc`, quy tắc trong `CLAUDE.md`. Gốc: 28 test. |
| M1 | 9/9: công cụ lỗi không làm sập agent, tách JSON, cất corpus 10T, sửa link, viết lại README. 34 test. |
| M2 | 8/8: `kind`/`params_b`, `AutoModelForMultimodalLM`, 4bit/8bit, QLoRA (`lora` + `4bit`), fp16 cho T4, lệnh `vram`. 54 test. |
| M3 | 7/7: adapter server local (urllib, chỉ mạng nội bộ, timeout, lỗi tiếng Việt), demo `--model`. 71 test. |
| M4 | 9/9: 3 preset, trộn tỉ lệ, `list-presets`, sửa lỗi hội thoại nhiều lượt và lỗi mất reasoning/context. Lỗi `datasets` lúc thoát đã xử lý. 85 test. |
| M5 | 8/8: 4 cách chấm, bộ đề 30 câu, lệnh eval, chặn trùng theo nội dung. Test cũ dựa vào lỗi cũ đã đổi fixture. 105 test. |
| M6 | 7/7: chạy thật trên CPU; thêm `max_steps`, `adapter_path`, `max_new_tokens`. 108 test. |
| M7 | 7/7: README (trạng thái, GPU smoke → light → primary, lộ trình đề xuất), ARCHITECTURE, test README khớp code. Sửa id `"NaN"` và secret-scan quét cả file bị `.gitignore`. 114 test, compileall và secret-scan sạch. |
| Gộp | PR #6 gộp vào `main` (`63bdc61`), PR #5 đã đóng. |
| Rà soát | Thống nhất sau khi gộp: lệnh trộn preset mặc định ghi vào `data/processed/hf_sft` (trùng chỗ train đọc; trước là `hf_mix`); `hf_sft.json` đọc streaming, limit 1000, có `language` và trạng thái giấy phép như preset; lệnh `vram` tìm cấu hình theo thư mục gốc repo; trợ giúp tiếng Việt cho mọi lệnh con của `local_ai.data`; lỗi `verify_math` bằng tiếng Việt; README ghi đủ thư mục và chạy lệnh từ thư mục gốc; thêm `tests/test_consistency.py`. Mô tả repo trên GitHub ("Train Từ Số 0") còn lệch hướng fine-tune: chủ repo tự sửa trong Settings. |
