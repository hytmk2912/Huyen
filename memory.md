# Bộ nhớ làm việc

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn, chỉ ghi những gì lượt sau cần biết. Kế hoạch và tiêu chí nằm trong `TASKS.md`.

## Mốc đang làm
- Chưa bắt đầu mốc nào. Mốc kế tiếp: **M1**.

## Việc dở
- **PR #4** (nhánh `claude/expand-model-training-repo-v2yzyr`) chưa gộp và **đi ngược một phần kế hoạch này**: PR xoá hẳn agent (trong khi M1 cần giữ và sửa agent) và xoá hẳn corpus (trong khi M1 cất corpus vào `archive/`). PR #4 cũng có phần dùng lại được:
  - sửa 3 lỗi dữ liệu (commit `2626c7d`), dùng cho M4 và M5;
  - QLoRA, ước lượng bộ nhớ và `pyproject.toml` (commit `59230b9`), dùng cho M2;
  - `local_ai/evaluation/suites.py`, dùng cho M5.

  Chủ repo cần quyết định đóng hay gộp PR #4.
- Yêu cầu đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B (bf16) mới được làm ở PR #4. `main` vẫn đang dùng Qwen2.5-Coder-7B, Qwen3-4B và Qwen2.5-0.5B.
- Việc chủ repo tự làm (AI không làm được):
  - đổi mật khẩu máy chủ cũ còn nằm trong lịch sử commit;
  - cân nhắc chuyển repo sang Private;
  - xoá nhánh `codex/build-autonomous-ai-system-architecture` (nhánh này đã gộp hết vào `main`).

## Lỗi gặp (chưa sửa)

| Lỗi | Nơi | Mốc sửa |
| --- | --- | --- |
| Công cụ chia 0 (`ZeroDivisionError`) hoặc sai cú pháp (`SyntaxError`) làm sập cả agent, vì `ToolRegistry.execute` chỉ bắt `TypeError` và `ValueError`. Đã tái hiện ngày 24/9. | `local_ai/tools/core.py` | M1 |
| `_parse_json_object` trả về `{}` khi JSON có chữ thừa, nằm trong khối ```` ```json ````, hoặc đứng sau `<think>`. Đã tái hiện ngày 24/9. | `local_ai/agents/loop.py` | M1 |
| Hội thoại nhiều lượt bị ghép sai (câu hỏi đầu ghép với câu trả lời cuối) và mất system prompt. | `local_ai/data/hub.py` | M4 |
| `sft.jsonl` bỏ mất reasoning và context. | `local_ai/data/core.py` (`prepare_format`) | M4 |
| Bước chặn rò rỉ eval chỉ so ID, nên câu y hệt nhưng khác ID vẫn lọt vào train. | `local_ai/data/core.py` (`build_dataset`) | M5 |
| Hugging Face gắn nhãn `image-text-to-text` cho model chính, nhưng adapter nạp bằng `AutoModelForCausalLM`. Chưa chạy thật để xác nhận. | `local_ai/models/adapters.py` | M2 |
| Trường `quantization` có khai báo nhưng không code nào đọc. | `local_ai/models/adapters.py` | M2 |
| Có khóa cấu hình không dùng trong `corpus_10t.json` và `smoke_real.json`. | `configs/datasets/` | M1 |
| Link cũ `huyenb2404-ops`. | `data/raw/seed_examples.jsonl`, `data/eval/seed_eval.jsonl` | M1 |

## Nhật ký các lượt

| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 2026-09-24 | Thiết lập nhiệm vụ 1 tuần | Tạo `TASKS.md`, `memory.md` và skill `lam-moc`; thêm quy tắc vào `CLAUDE.md`; thêm `.env` và `.venv/` vào `.gitignore`. Kiểm tra gốc trên `main`: 28 test chạy qua (1 bỏ qua), compileall và secret-scan sạch. Chưa làm mốc nào. |
