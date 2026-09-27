# Nhật ký chi tiết tuần 4

Mỗi lượt một dòng; `memory.md` chỉ giữ dòng ngắn.

| Ngày | Lượt | Chi tiết |
| --- | --- | --- |
| 27/9 | Mở tuần 4 (chủ repo yêu cầu, không phải mốc) | `TASKS.md` tuần 3 chép nguyên văn (`cmp` trùng khớp) sang `archive/tasks-tuan-3.md`; `TASKS.md` mới cho M23–M27 theo tiêu chí chủ repo đặt, bắt đầu từ `main` `db0f9d4` (338 test). `memory.md` đổi sang TUẦN 4, giữ nguyên mục "Việc chủ repo tự làm"; bản cuối memory tuần 3 chép nguyên văn vào cuối `archive/memory-tuan-3.md`. Skill `lam-moc` và `CLAUDE.md` đổi chỗ nhắc tuần 3 sang tuần 4. Test đọc kế hoạch và memory tuần 3 (M21, rà soát tuần 3) đổi chỗ đọc sang file archive, giữ nguyên ý kiểm tra. |
| 27/9 | M23 (xong) | `openai_compatible.py`: payload thêm `"max_tokens": config.max_new_tokens`. Test mới `tests/test_m23_max_tokens.py` (4 test) dùng server HTTP giả của test M3: mặc định 512, giá trị tự đặt, lệnh chấm (giá trị trong danh sách model 200, rồi `--max-new-tokens 32`), các mục server trong `platform.json`. Kiểm chứng test bắt được lỗi: tạm bỏ sửa (git stash) thì 4 test lỗi. Test M3 so nguyên payload: thêm `"max_tokens": 512` (yêu cầu mới của M23). Không lỗi mới. 342 test, KẾT QUẢ: XANH. |
