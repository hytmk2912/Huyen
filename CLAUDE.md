# Quy ước cho repo này

Đọc thêm `memory.md`: các quyết định đã chốt. Không làm ngược các quyết định đó nếu chủ repo chưa đồng ý.

## Phạm vi (quan trọng nhất)
Repo này **chỉ** làm một việc: fine-tune model Qwen3 (Huihui) có sẵn.

    dataset Hugging Face → kiểm tra/loại trùng/chặn rò rỉ eval → sft.jsonl → fine-tune LoRA/QLoRA → đánh giá sau train

- Không thêm: agent, công cụ cho agent, pretrain/corpus quy mô lớn, thu thập dữ liệu web, giao dịch, xuất GGUF, huấn luyện phân tán.
- Không tự thêm tính năng hay tự mở rộng lộ trình. Chỉ làm đúng việc chủ repo yêu cầu; việc nằm ngoài phạm vi thì hỏi trước.
- Không tự hẹn giờ chạy lặp để thêm việc.
- Sửa lỗi và làm gọn code trong phạm vi trên thì được làm.

## Ngôn ngữ
Chủ repo không đọc được tiếng Anh. Mọi nội dung đẩy lên GitHub phải viết bằng **tiếng Việt**:
- commit message, tiêu đề và mô tả pull request, bình luận trên GitHub;
- tài liệu (`README.md`, `docs/`, `memory.md`), chú thích `_comment` trong file cấu hình;
- docstring, comment trong code, phần trợ giúp (`help=`) của lệnh và thông báo lỗi dành cho người dùng.

Giữ nguyên tiếng Anh cho: tên hàm/biến/lớp, khóa JSON, giá trị trạng thái (ví dụ `"skipped"`), thuật ngữ kỹ thuật phổ biến (LoRA, checkpoint, dataset, token...) và các câu lệnh (prompt) gửi cho model.

## Bảo mật
- `HF_TOKEN` và mọi khóa chỉ để trong biến môi trường hoặc file `.env` (đã có trong `.gitignore`). Không bao giờ ghi khóa, mật khẩu, IP máy chủ vào code, cấu hình hay script.

## Kiểm tra trước khi đẩy
```bash
python -m unittest discover -s tests -v
python -m compileall -q local_ai
python -m local_ai.data secret-scan
ruff format local_ai tests && ruff check local_ai tests
```
Mục cần GPU hoặc khóa thật: code phải tự bỏ qua (`"skipped"`) khi thiếu, và không được ghi là đã chạy thật khi chưa chạy.
