# Quy ước cho repo này

## Nhiệm vụ tuần
**Đọc `memory.md` trước khi làm việc.**

Kế hoạch nằm trong `TASKS.md`: tuần 2 (M8–M14) ở đầu file, tuần 1 (M1–M7, đã xong) ở dưới. Mỗi mốc có tiêu chí xong. Mốc đang làm, việc dở và lỗi gặp nằm trong `memory.md` (nhật ký tuần 1: `archive/memory-tuan-1.md`). Làm mốc bằng skill `lam-moc` (`.claude/skills/lam-moc/SKILL.md`).
- Mỗi lượt làm tối đa **1 mốc**, theo thứ tự trong `TASKS.md`.
- Chỉ đánh dấu một mốc là xong khi test xanh (kèm `compileall` và `secret-scan`) và mọi tiêu chí của mốc đã đạt.
- Không dùng API trả phí (OpenAI, Anthropic, Gemini...). Chỉ chạy model cục bộ hoặc qua server local.
- Không tải trọng số model hay dataset lớn. Test không dùng mạng, không cần GPU (dùng model tí hon khởi tạo ngẫu nhiên, loader giả). Dataset thật chỉ đọc streaming với `limit` nhỏ.
- Token chỉ đọc từ biến môi trường hoặc Colab Secrets; không ghi khóa vào repo hay notebook.
- Notebook: lưu không kèm output, ghim phiên bản thư viện, mỗi ô có chú thích tiếng Việt. Không chạy notebook trong môi trường phát triển; chỉ kiểm tra hợp lệ và chạy thử lệnh của nó bằng `--dry-run`.
- Code bỏ đi thì cất vào `archive/`. Không sửa test cũ chỉ để cho xanh.
- Chỉ làm việc có trong `TASKS.md`. Việc khác thì ghi vào "Việc dở" trong `memory.md` và hỏi chủ repo trước. Không tự hẹn giờ chạy lặp.

## Ngôn ngữ
Chủ repo không đọc được tiếng Anh. Mọi nội dung đẩy lên GitHub phải viết bằng **tiếng Việt**:
- commit message, tiêu đề và mô tả pull request, bình luận trên GitHub;
- tài liệu (`README.md`, `docs/`), chú thích `_comment` trong file cấu hình;
- docstring, comment trong code, phần trợ giúp (`help=`) của lệnh và thông báo lỗi dành cho người dùng.

Giữ nguyên tiếng Anh cho: tên hàm/biến/lớp, khóa JSON, giá trị trạng thái (ví dụ `"skipped"`), thuật ngữ kỹ thuật phổ biến (LoRA, checkpoint, dataset, token...) và các câu lệnh (prompt) gửi cho model.

## Kiểm tra trước khi đẩy
```bash
python -m unittest discover -s tests -v
python -m compileall -q local_ai
python -m local_ai.data secret-scan
```
