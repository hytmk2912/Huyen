# Bộ nhớ làm việc

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng), chỉ ghi những gì lượt sau cần biết. Kế hoạch và tiêu chí nằm trong `TASKS.md`.

## Checklist
- [x] M1 Sửa lỗi + dọn repo (xong 24/9)
- [x] M2 Nạp model đúng loại + nén 4-bit (xong 24/9)
- [ ] M3 Adapter server local kiểu OpenAI
- [ ] M4 Preset dataset
- [ ] M5 Đánh giá (eval)
- [ ] M6 Chạy thật trên CPU
- [ ] M7 Tổng kết

## Mốc đang làm
- Không có mốc dở. Mốc kế tiếp: **M3**.

## Việc dở
- **PR #4** (nhánh `claude/expand-model-training-repo-v2yzyr`) chưa gộp và **đi ngược một phần kế hoạch này** (xoá hẳn agent và corpus thay vì sửa agent và cất corpus vào `archive/`). Phần còn dùng lại được: sửa 3 lỗi dữ liệu (commit `2626c7d`, cho M4, M5) và `local_ai/evaluation/suites.py` (cho M5). Chủ repo cần quyết định đóng hay gộp PR #4.
- **PR #5** (nhánh `claude/nhiem-vu-1-tuan`) chỉ có commit thiết lập, đã nằm trong PR #6; gộp PR #6 thì có thể đóng PR #5.
- Nhánh làm việc là `claude/nhiem-vu-tuan` (PR #6). Phiên chạy tự động được gán nhánh `claude/nhiem-vu-tuan-<mã>`, nhưng lượt vẫn đẩy lên `claude/nhiem-vu-tuan` theo yêu cầu nhiệm vụ.
- Yêu cầu đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B (bf16) mới có ở PR #4; `main` vẫn dùng Qwen2.5-Coder-7B, Qwen3-4B, Qwen2.5-0.5B.
- Việc chủ repo tự làm (AI không làm được): đổi mật khẩu máy chủ cũ còn trong lịch sử commit; cân nhắc chuyển repo sang Private; xoá nhánh `codex/build-autonomous-ai-system-architecture` (đã gộp hết vào `main`).

## Lỗi gặp (chưa sửa)

| Lỗi | Nơi | Mốc sửa |
| --- | --- | --- |
| Hội thoại nhiều lượt bị ghép sai (câu hỏi đầu ghép với câu trả lời cuối) và mất system prompt. | `local_ai/data/hub.py` | M4 |
| `sft.jsonl` bỏ mất reasoning và context. | `local_ai/data/core.py` (`prepare_format`) | M4 |
| Bước chặn rò rỉ eval chỉ so ID, nên câu y hệt nhưng khác ID vẫn lọt vào train. | `local_ai/data/core.py` (`build_dataset`) | M5 |
| Chưa chạy thật: nạp model multimodal, nén 4bit và QLoRA mới chỉ test bằng module giả. `target_modules: "all-linear"` có thể gắn LoRA cả vào phần xử lý ảnh của model chính; cần xem lại khi có GPU. | `local_ai/models/adapters.py`, `local_ai/training/finetune.py` | M6 / khi có GPU |

## Nhật ký các lượt

| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 24/9 | Thiết lập | Tạo `TASKS.md`, `memory.md`, skill `lam-moc`; thêm quy tắc vào `CLAUDE.md`. Gốc trên `main`: 28 test qua (1 bỏ qua). |
| 24/9 | M1 | Xong 9/9. Công cụ lỗi không làm sập agent; `extract_json_object` tách JSON (chữ thừa, khối code, `<think>`); cất corpus 10T vào `archive/corpus-10t/`; tách `secret-scan` sang `secrets.py`; sửa link cũ; viết lại README. 34 test qua. |
| 24/9 | M2 | Xong 8/8. `platform.json` thêm `kind`, `params_b` (số liệu lấy từ Hugging Face: model chính `AutoModelForMultimodalLM`, 27,78 tỷ). Adapter: multimodal → `AutoProcessor` + `AutoModelForMultimodalLM` (dự phòng `AutoModelForImageTextToText`), gửi được ảnh; nén `4bit`/`8bit` qua bitsandbytes; GPU không bf16 tự chuyển fp16. QLoRA = `method: "lora"` + `quantization: "4bit"` (giữ nguyên test cũ yêu cầu từ chối `method: "qlora"`), mẫu `configs/training/qlora_primary.json`. Lệnh `python -m local_ai.models.vram`. Cập nhật skill `lam-moc` theo quy trình mới, thêm "Đọc memory.md" vào `CLAUDE.md`. Kiểm tra: 54 test qua, compileall và secret-scan sạch. Lỗi gặp: không có. Tiếp theo: M3. |
