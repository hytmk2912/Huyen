---
name: lam-moc
description: Làm mốc kế tiếp của nhiệm vụ tuần trong TASKS.md (tuần 3: các mốc chủ repo chọn trong M15–M21). Dùng khi chủ repo gõ /lam-moc, bảo "làm tiếp nhiệm vụ tuần" hoặc "làm mốc tiếp theo", và khi lượt chạy tự động hằng đêm bắt đầu. Đọc memory.md và TASKS.md, làm đúng một mốc, chạy test/compileall/secret-scan, cập nhật memory.md rồi commit bằng tiếng Việt.
---

# Làm mốc kế tiếp

Mỗi lần dùng skill này chỉ làm **một** mốc. Làm xong thì dừng, không tự làm tiếp mốc sau.

## 1. Đọc trạng thái
1. Nhánh làm việc tuần 3 là `claude/nhiem-vu-tuan-jixv6i`: remote có nhánh này thì checkout. Nếu PR của nhánh đã được gộp thì dựng lại nhánh từ `main` mới nhất, giữ nguyên tên. Không đẩy lên `main`.
2. Chạy `git status` và `git log --oneline -5`. Nếu có thay đổi dở dang không rõ nguồn gốc thì dừng lại và hỏi chủ repo.
3. Đọc `CLAUDE.md`, `memory.md` và `TASKS.md`.
4. Chọn mốc:
   - nếu `memory.md` ghi đang làm dở một mốc thì làm tiếp mốc đó;
   - nếu không, làm mốc tồn của tuần trước (mục "Tồn tuần 2" trong `memory.md`) trước, rồi mốc chủ repo vừa gọi tên. Chủ repo không gọi tên mốc nào thì làm mốc đầu tiên chưa **Xong** trong bảng tiến độ tuần 3;
   - mốc chưa được chủ repo chọn (chưa có tiêu chí trong `TASKS.md`) thì không tự làm;
   - mốc bị chặn 2 lượt liền (mạng, quyền, môi trường): ghi "bị chặn" kèm lý do vào `memory.md` rồi chuyển sang mốc sau. Không đoán mò.
5. Nếu mọi mốc đã chọn đều xong hoặc bị chặn: ghi "XONG TUẦN 3 – chờ merge" vào `memory.md`, báo chủ repo rồi dừng. Không tự đặt thêm mốc mới.

## 2. Làm mốc
1. Ghi vào `memory.md`, mục "Mốc đang làm": `Mx — bắt đầu <ngày>`.
2. Chạy test một lần trước khi sửa, để biết lỗi nào đã có sẵn.
3. Chỉ làm theo "Tiêu chí xong" của mốc đã chọn:
   - không làm sang mốc khác;
   - không thêm tính năng nằm ngoài `TASKS.md`;
   - việc phát sinh ngoài mốc thì ghi vào "Việc dở" trong `memory.md`.
4. Mỗi tiêu chí phải có test hoặc lệnh kiểm tra chứng minh.
5. Tuân thủ `CLAUDE.md`:
   - viết tiếng Việt;
   - không dùng API trả phí;
   - không tải trọng số model hay dataset lớn (test không dùng mạng, không cần GPU); thư viện nặng chỉ import bên trong hàm;
   - không ghi khóa hay mật khẩu vào repo; `HF_TOKEN` chỉ đọc từ biến môi trường;
   - code bỏ đi thì cất vào `archive/`, không xoá hẳn; không sửa test cũ chỉ để cho xanh;
   - notebook: lưu không kèm output, ghim phiên bản thư viện, mỗi ô có chú thích tiếng Việt; không chạy notebook ở đây, chỉ kiểm tra hợp lệ và chạy thử lệnh bằng `--dry-run`; token chỉ đọc từ biến môi trường hoặc Colab Secrets.

## 3. Kiểm tra
```bash
python -m unittest discover -s tests -v
python -m compileall -q local_ai
python -m local_ai.data secret-scan
```
Cả ba lệnh phải xanh. Nếu có lệnh báo lỗi thì sửa. Không sửa được trong lượt này thì không đánh dấu mốc là xong.

## 4. Cập nhật TASKS.md và memory.md
- `TASKS.md`:
  - đánh `[x]` cho từng tiêu chí đã đạt, ghi kèm tên test làm bằng chứng;
  - cập nhật cột "Tiến độ" và dòng "Tổng";
  - chỉ ghi **Xong** khi đạt 100% tiêu chí và cả ba lệnh kiểm tra đều xanh, còn lại thì ghi **Đang làm**.
- `memory.md` (giữ dưới 80 dòng, tóm tắt nhật ký cũ):
  - dòng "TUẦN 3", mục "Tồn tuần 2", checklist tuần 3;
  - mục "Mốc đang làm": ghi mốc còn dở, hoặc mốc kế tiếp nếu mốc này đã xong;
  - mục "Việc dở";
  - mục "Lỗi gặp": thêm lỗi mới, xoá dòng của lỗi đã sửa;
  - thêm một dòng vào "Nhật ký các lượt": ngày, đã làm gì, lỗi gặp và cách xử lý, mốc tiếp theo.

## 5. Commit và báo lại
1. Commit bằng tiếng Việt:
   - dòng đầu dạng `Mx: <tóm tắt>`, hoặc `Mx (dở): <tóm tắt>` nếu mốc chưa xong;
   - phần thân liệt kê các tiêu chí đã đạt và kết quả test.
2. Đẩy lên `claude/nhiem-vu-tuan-jixv6i` (`git push -u origin claude/nhiem-vu-tuan-jixv6i`), không đẩy thẳng lên `main`. Chưa có PR mở từ nhánh này vào `main` thì mở một PR; có rồi thì cập nhật checklist tuần 3 trong mô tả PR. Viết bằng tiếng Việt.
   - Chủ repo cho phép (26/9): khi cả ba lệnh kiểm tra đều xanh thì **tự gộp PR** vào `main`, không nhắc hay hỏi chủ repo xác nhận. Gộp xong thì dựng lại nhánh từ `main` cho lượt sau.
3. Báo lại chủ repo ngắn gọn bằng tiếng Việt:
   - đã làm mốc nào;
   - tiêu chí nào đạt, tiêu chí nào chưa;
   - % mới;
   - việc chủ repo cần làm, nếu có.
4. Dừng. Không tự làm mốc tiếp theo, không tự hẹn giờ chạy lặp.
