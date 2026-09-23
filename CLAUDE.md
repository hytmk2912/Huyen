# Quy ước cho repo này

## Ngôn ngữ
Chủ repo không đọc được tiếng Anh. Mọi nội dung đẩy lên GitHub phải viết bằng **tiếng Việt**:
- commit message, tiêu đề và mô tả pull request, bình luận trên GitHub;
- tài liệu (`README.md`, `docs/`), chú thích `_comment` trong file cấu hình;
- docstring, comment trong code, phần trợ giúp (`help=`) của lệnh và thông báo lỗi dành cho người dùng.

Giữ nguyên tiếng Anh cho: tên hàm/biến/lớp, khóa JSON, giá trị trạng thái (ví dụ `"skipped"`), thuật ngữ kỹ thuật phổ biến (LoRA, checkpoint, dataset, token...) và các câu lệnh (prompt) gửi cho model.

## Lộ trình (README.md, mục "Lộ trình")
- Luôn giữ đúng 5 mục trong "Đang làm". Làm xong một mục thì chuyển xuống "Đã hoàn thành" (ghi ngắn gọn đã làm gì) và thêm ngay một mục mới chưa làm, phù hợp hướng dự án.
- Mục cần phần cứng/khóa truy cập thật: làm phần code và test (tự bỏ qua khi thiếu GPU/thư viện), phần chạy thật ghi vào "Việc cần chạy trên máy thật". Không ghi là đã chạy thật khi chưa chạy.

## Kiểm tra trước khi đẩy
```bash
python -m unittest discover -s tests -v
python -m compileall -q local_ai
python -m local_ai.data secret-scan
```
