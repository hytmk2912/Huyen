# Quy ước cho repo này

> Giữ file này dưới 200 dòng và chỉ ghi quy tắc. Trạng thái công việc ghi ở `memory.md` và `TASKS.md`.

## Nhiệm vụ tuần
**Đọc `memory.md` trước khi làm việc.**

Kế hoạch tuần 3 (chủ repo chọn từng mốc trong M15–M21) nằm trong `TASKS.md`; tuần 1–2 (M1–M14, đều đã xong) chép nguyên văn ở `archive/tasks-tuan-1-2.md`. Mỗi mốc có tiêu chí xong. Mốc đang làm, việc dở và lỗi còn tồn nằm trong `memory.md` (tối đa 5.000 ký tự; nhật ký chi tiết trong `archive/`). Làm mốc bằng skill `lam-moc` (`.claude/skills/lam-moc/SKILL.md`).
- Mỗi lượt làm tối đa **1 mốc**: mốc chủ repo gọi tên, hoặc mốc đầu tiên chưa xong trong `TASKS.md`.
- Chỉ đánh dấu một mốc là xong, và chỉ tự gộp PR, khi `python -m local_ai.check` (không có `--allow-skip`) xanh và mọi tiêu chí của mốc đã đạt. Lệnh này chạy test, `compileall` và `secret-scan`; test bị bỏ qua vì thiếu thư viện cũng tính là chưa xanh. Thiếu thư viện thì cài theo README (torch bản CPU) rồi chạy lại.
- Không dùng API trả phí (OpenAI, Anthropic, Gemini...). Chỉ chạy model cục bộ hoặc qua server local.
- Không tải trọng số model hay dataset lớn. Test không dùng mạng, không cần GPU (dùng model tí hon khởi tạo ngẫu nhiên, loader giả). Dataset thật chỉ đọc streaming với `limit` nhỏ.
- Token chỉ đọc từ biến môi trường hoặc Colab Secrets; không ghi khóa vào repo hay notebook.
- Notebook: lưu không kèm output, ghim phiên bản thư viện, mỗi ô có chú thích tiếng Việt. Không chạy notebook trong môi trường phát triển; chỉ kiểm tra hợp lệ và chạy thử lệnh của nó bằng `--dry-run`.
- Code bỏ đi thì cất vào `archive/`. Không sửa test cũ chỉ để cho xanh.
- Chỉ làm việc có trong `TASKS.md`. Việc khác thì ghi vào "Việc dở" trong `memory.md` và hỏi chủ repo trước. Không tự hẹn giờ chạy lặp.

## Khi nào phải dừng hỏi chủ repo
Việc trong phạm vi mốc (sửa code, chạy test, commit, mở PR, tự gộp PR khi check xanh) thì cứ làm, không xin xác nhận từng bước. Chỉ dừng và hỏi trước khi:
- xoá hẳn file, thư mục hay nhánh (code bỏ đi vẫn cất vào `archive/`);
- `git push --force`, `git reset --hard`, viết lại lịch sử commit;
- ghi đè dữ liệu đã có: số đo thật `tests/fixtures/measurements/that_*.json`, dữ liệu trên Hugging Face Hub;
- nới lỏng sandbox hay TerminalTool (cho chạy thêm lệnh, lộ thêm biến môi trường).

Khi hỏi: một dòng nói định làm gì, hậu quả, cách hoàn tác. Chủ repo đã yêu cầu rõ việc đó thì coi như đã duyệt. Lượt chạy tự động không có ai trả lời: không làm, ghi vào "Việc dở" trong `memory.md`.

## Chống quên khi nén ngữ cảnh
- Sau mỗi lần nén ngữ cảnh (`/compact` hoặc tự động): đọc lại `memory.md`, `TASKS.md`, chạy `git status` rồi mới làm tiếp.
- Đạt tiêu chí nào thì đánh `[x]` trong `TASKS.md` (kèm tên test) và ghi bước kế tiếp vào "Việc dở" của `memory.md` **ngay lúc đó**, không đợi cuối lượt.

## Chia việc cho subagent
- Chỉ chia nhỏ việc của mốc đang làm, không dùng để làm song song nhiều mốc.
- Chỉ giao việc độc lập: mỗi subagent một nhóm file riêng, không để hai subagent sửa cùng một file.
- Giao việc phải ghi rõ: tiêu chí cần đạt, file được sửa, test phải chạy.
- Subagent phải báo lại đủ bằng chứng, thiếu thì coi như chưa xong:
  - `git diff --stat` các file mình sửa (file nào, bao nhiêu dòng);
  - lệnh test đã chạy và dòng tổng kết (số test đạt, số lỗi);
  - việc chưa làm hoặc rủi ro.
- Số dòng chỉ cho biết phạm vi thay đổi; đúng hay sai dựa vào kết quả test.
- Gộp xong, agent chính tự chạy `python -m local_ai.check`, không tin báo cáo suông.

## Ngôn ngữ
Chủ repo không đọc được tiếng Anh. Mọi nội dung đẩy lên GitHub phải viết bằng **tiếng Việt**:
- commit message, tiêu đề và mô tả pull request, bình luận trên GitHub;
- tài liệu (`README.md`, `docs/`), chú thích `_comment` trong file cấu hình;
- docstring, comment trong code, phần trợ giúp (`help=`) của lệnh và thông báo lỗi dành cho người dùng.

Giữ nguyên tiếng Anh cho: tên hàm/biến/lớp, khóa JSON, giá trị trạng thái (ví dụ `"skipped"`), thuật ngữ kỹ thuật phổ biến (LoRA, checkpoint, dataset, token...) và các câu lệnh (prompt) gửi cho model.

## Kiểm tra trước khi đẩy
```bash
python -m local_ai.check    # test + compileall + secret-scan; test bị bỏ qua vì thiếu thư viện thì không xanh
```
Báo xong (với chủ repo hay trong `memory.md`) phải kèm số test và dòng `KẾT QUẢ:` mà lệnh in ra.
