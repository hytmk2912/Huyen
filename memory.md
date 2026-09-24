# Bộ nhớ làm việc

TUẦN 2

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng). Kế hoạch và tiêu chí: `TASKS.md` (tuần 2 ở đầu file). Nhật ký tuần 1: `archive/memory-tuan-1.md`.

## Tồn tuần 1
- Không có mốc tồn: M1–M7 đều xong, đã gộp vào `main` (PR #6 `63bdc61`, PR #7 `09c93f3`).
- Việc cần GPU hoặc server thật (không phải mốc): chạy smoke → light → primary trên GPU; thử adapter server local với Ollama/llama.cpp thật.

## Checklist tuần 2
- [x] M8 Gộp repo Agent, phần 1: đưa code runtime vào (xong 24/9)
- [ ] M9 Gộp repo Agent, phần 2: tool chạy lệnh an toàn
- [ ] M10 Notebook train trên Colab free (0.5B)
- [ ] M11 Colab cho Qwen3-4B + hướng dẫn iPhone
- [ ] M12 Chất lượng dữ liệu
- [ ] M13 Agent chạy model thật trên Colab
- [ ] M14 Tổng kết tuần 2

## Mốc đang làm
- Không có mốc dở. Mốc kế tiếp: **M9** (TerminalTool, allowlist trong `configs/tools/terminal.json`, gateway tắt mặc định; xem mục "Phần để mốc M9" và "Rủi ro bảo mật" trong `docs/GOP_AGENT.md`).

## Việc dở
- Nhánh làm việc: `claude/nhiem-vu-tuan-2` (tạo từ `main` tại `09c93f3`).
- PR #4 (`claude/expand-model-training-repo-v2yzyr`) vẫn mở, chờ chủ repo quyết định.
- Repo Agent: `hytmk2912/Agent` không truy cập được từ phiên (riêng tư hoặc chưa cấp quyền); đã lấy bản công khai `huyenytmk2912/agent` tại `78a3e25`. Nếu tên mới có code mới hơn thì cần chủ repo cấp quyền để gộp thêm. Sau khi gộp xong (M9), chủ repo archive repo Agent cũ.
- Việc chủ repo tự làm: sửa mô tả repo trên GitHub ("Train Từ Số 0"); đổi mật khẩu máy chủ cũ còn trong lịch sử commit; cân nhắc chuyển repo sang Private; đọc lại giấy phép 3 dataset preset.

## Lỗi còn tồn (từ tuần 1)
| Lỗi | Nơi |
| --- | --- |
| Chưa chạy thật trên GPU: QLoRA, model ảnh + chữ; `target_modules: "all-linear"` có thể gắn LoRA vào phần xử lý ảnh. | `local_ai/models/adapters.py`, `local_ai/training/finetune.py` |
| transformers 5.17 cảnh báo `torch_dtype` đã cũ, nên đổi sang `dtype`. | `adapters.py`, `finetune.py` |

## Nhật ký tuần 2
| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 24/9 | Thiết lập | Tạo nhánh `claude/nhiem-vu-tuan-2` từ `main` (`09c93f3`); chuyển nhật ký tuần 1 sang `archive/memory-tuan-1.md`; thêm kế hoạch M8–M14 vào đầu `TASKS.md`; cập nhật skill `lam-moc` và `CLAUDE.md` cho tuần 2 (quy tắc notebook, Colab Secrets). |
| 24/9 | M8 | Xong 4/4. Clone `huyenytmk2912/agent` (`78a3e25`, 4 file, không có test). Viết lại runtime bằng thư viện chuẩn trong `local_ai/runtime/` (`run_command` chạy lệnh dạng list không qua shell, dừng cả nhóm tiến trình khi quá giờ; `JobQueue` có giới hạn; `require_token` so sánh an toàn). Cất code gốc ở `archive/agent-goc/`. `docs/GOP_AGENT.md` ghi 7 rủi ro bảo mật của bản gốc. Lỗi gặp: test tìm chữ `shell=True` bắt nhầm docstring, đổi sang kiểm tra bằng AST; lần chạy đầu 57 giây do phiên mới (lần sau 10 giây). 133 test qua, compileall và secret-scan sạch. Tiếp theo: M9. |
