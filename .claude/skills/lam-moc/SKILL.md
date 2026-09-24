---
name: lam-moc
description: Làm mốc kế tiếp của nhiệm vụ 1 tuần trong TASKS.md. Dùng khi chủ repo gõ /lam-moc hoặc bảo "làm mốc tiếp theo". Đọc memory.md và TASKS.md, làm đúng một mốc, chạy test/compileall/secret-scan, cập nhật memory.md rồi commit bằng tiếng Việt.
---

# Làm mốc kế tiếp

Mỗi lần dùng skill này chỉ làm **một** mốc. Làm xong thì dừng, không tự làm tiếp mốc sau.

## 1. Đọc trạng thái
1. Chạy `git status` và `git log --oneline -5`. Nếu có thay đổi dở dang không rõ nguồn gốc thì dừng lại và hỏi chủ repo.
2. Đọc `CLAUDE.md`, `memory.md` và `TASKS.md`.
3. Chọn mốc:
   - nếu `memory.md` ghi đang làm dở một mốc thì làm tiếp mốc đó;
   - nếu không, chọn mốc đầu tiên (theo thứ tự M1 → M7) chưa **Xong** trong bảng tiến độ.
4. Nếu cả 7 mốc đã xong thì báo chủ repo. Không tự đặt thêm mốc mới.

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
   - không tải trọng số model hay dataset lớn (test không dùng mạng);
   - không ghi khóa hay mật khẩu vào repo.

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
- `memory.md`:
  - mục "Mốc đang làm": ghi mốc còn dở, hoặc mốc kế tiếp nếu mốc này đã xong;
  - mục "Việc dở";
  - mục "Lỗi gặp": thêm lỗi mới, xoá dòng của lỗi đã sửa;
  - thêm một dòng vào "Nhật ký các lượt".

## 5. Commit và báo lại
1. Commit bằng tiếng Việt:
   - dòng đầu dạng `Mx: <tóm tắt>`, hoặc `Mx (dở): <tóm tắt>` nếu mốc chưa xong;
   - phần thân liệt kê các tiêu chí đã đạt và kết quả test.
2. Đẩy lên nhánh làm việc của phiên, không đẩy thẳng lên `main`. Cập nhật mô tả PR bằng tiếng Việt.
3. Báo lại chủ repo ngắn gọn bằng tiếng Việt:
   - đã làm mốc nào;
   - tiêu chí nào đạt, tiêu chí nào chưa;
   - % mới;
   - việc chủ repo cần làm, nếu có.
4. Dừng. Không tự làm mốc tiếp theo, không tự hẹn giờ chạy lặp.
