# Bộ nhớ làm việc

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn, chỉ ghi những gì lượt sau cần biết. Kế hoạch và tiêu chí nằm trong `TASKS.md`.

## Mốc đang làm
- M1 xong ngày 2026-09-24. Mốc kế tiếp: **M2**.

## Việc dở
- **PR #4** (nhánh `claude/expand-model-training-repo-v2yzyr`) chưa gộp và **đi ngược một phần kế hoạch này**: PR xoá hẳn agent (trong khi M1 cần giữ và sửa agent) và xoá hẳn corpus (trong khi M1 cất corpus vào `archive/`). PR #4 cũng có phần dùng lại được:
  - sửa 3 lỗi dữ liệu (commit `2626c7d`), dùng cho M4 và M5;
  - QLoRA, ước lượng bộ nhớ và `pyproject.toml` (commit `59230b9`), dùng cho M2;
  - `local_ai/evaluation/suites.py`, dùng cho M5.

  Chủ repo cần quyết định đóng hay gộp PR #4.
- Nhánh làm việc hiện tại là `claude/nhiem-vu-tuan` (tạo từ `main`, có kèm commit thiết lập của PR #5). Nếu PR của nhánh này được gộp thì PR #5 trở nên thừa và có thể đóng.
- Yêu cầu đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B (bf16) mới được làm ở PR #4. `main` vẫn đang dùng Qwen2.5-Coder-7B, Qwen3-4B và Qwen2.5-0.5B.
- Việc chủ repo tự làm (AI không làm được):
  - đổi mật khẩu máy chủ cũ còn nằm trong lịch sử commit;
  - cân nhắc chuyển repo sang Private;
  - xoá nhánh `codex/build-autonomous-ai-system-architecture` (nhánh này đã gộp hết vào `main`).

## Lỗi gặp (chưa sửa)

| Lỗi | Nơi | Mốc sửa |
| --- | --- | --- |
| Hội thoại nhiều lượt bị ghép sai (câu hỏi đầu ghép với câu trả lời cuối) và mất system prompt. | `local_ai/data/hub.py` | M4 |
| `sft.jsonl` bỏ mất reasoning và context. | `local_ai/data/core.py` (`prepare_format`) | M4 |
| Bước chặn rò rỉ eval chỉ so ID, nên câu y hệt nhưng khác ID vẫn lọt vào train. | `local_ai/data/core.py` (`build_dataset`) | M5 |
| Hugging Face gắn nhãn `image-text-to-text` cho model chính, nhưng adapter nạp bằng `AutoModelForCausalLM`. Chưa chạy thật để xác nhận. | `local_ai/models/adapters.py` | M2 |
| Trường `quantization` có khai báo nhưng không code nào đọc. | `local_ai/models/adapters.py` | M2 |

## Nhật ký các lượt

| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 2026-09-24 | Thiết lập nhiệm vụ 1 tuần | Tạo `TASKS.md`, `memory.md` và skill `lam-moc`; thêm quy tắc vào `CLAUDE.md`; thêm `.env` và `.venv/` vào `.gitignore`. Kiểm tra gốc trên `main`: 28 test chạy qua (1 bỏ qua), compileall và secret-scan sạch. Chưa làm mốc nào. |
| 2026-09-24 | M1 | Xong 9/9 tiêu chí. Công cụ lỗi bất kỳ trả kết quả thất bại, không làm sập agent. Thêm `extract_json_object` để tách JSON (chịu chữ thừa, khối code, `<think>`). Cất corpus 10T vào `archive/corpus-10t/`, tách `secret-scan` sang `local_ai/data/secrets.py`, sửa link cũ, viết lại README. Kiểm tra: 34 test chạy qua, compileall và secret-scan sạch. |
