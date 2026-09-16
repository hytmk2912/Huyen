# Việc Cần Làm — Trước Và Sau Khi Có Thiết Bị

## Ai làm gì

**AI (Claude) lo:**
- Viết/sửa toàn bộ script, chọn model/dataset/config (đã xong — xem `README.md`)
- Khi có thiết bị: đưa đúng lệnh cần dán, đọc lỗi phát sinh và sửa, đọc kết quả train và đề xuất bước tiếp theo
- Cập nhật file này mỗi khi có thay đổi

**Bạn chỉ cần:**
1. Có VPS/máy tính (Ubuntu) — cần nhiều RAM CPU + ổ NVMe/SSD nhanh (không chỉ VRAM GPU, vì dùng kỹ thuật ZeRO-Infinity)
2. Dán script trong README.md vào máy, chạy
3. Copy nguyên lỗi (nếu có) hoặc kết quả gửi cho AI (trong đoạn chat mới) — AI xử lý tiếp
Không cần tự sửa code, tự tìm hiểu thêm gì khác.

## Cần gì — trạng thái hiện tại

| Thứ cần | Trạng thái | Ai lo |
|---|---|---|
| GitHub repo | ✅ Đã có (repo này) | Bạn |
| Personal Access Token (để AI ghi vào repo) | ⚠️ Token cũ có thể đã bị xoá — nếu AI cần ghi tiếp mà báo lỗi quyền, tạo token mới gửi lại | Bạn |
| VPS/máy tính (Ubuntu, nhiều RAM + ổ NVMe nhanh) | ❌ Chưa có | Bạn |
| Cách vào VPS từ iPhone | Nếu nơi thuê VPS có "Web Console/Terminal" trên trình duyệt thì dùng luôn, không cần cài gì. Nếu không, cần 1 app SSH miễn phí (vd Termius bản free) | Bạn |
| GPU thuê theo giờ (từ 16GB VRAM trở lên) | ❌ Chưa thuê (chỉ thuê đúng lúc train) | Bạn |
| Script cài đặt + train (nanoGPT + DeepSpeed) | ✅ Có sẵn trong `README.md` | AI |
| Bộ dữ liệu (~30 repo code + FineWeb-Edu + Wikipedia + toán/khoa học) | Script tự tải khi chạy — free, tự kiểm tra đủ dung lượng đĩa trước khi tải | AI |
| PyTorch, tiktoken, DeepSpeed, git | Script tự cài khi chạy | AI (viết script) |

## Lưu ý quan trọng — phần chưa kiểm chứng được 100%

Máy AI dùng để soạn không đủ ổ đĩa cài PyTorch và không có GPU, nên đoạn tải dữ liệu + train **chưa được chạy thử thật**. Code viết đúng theo tài liệu chính thức của nanoGPT và Hugging Face nên khả năng chạy đúng khá cao, nhưng nếu lúc chạy thật bị lỗi ở bước nào, cứ copy nguyên lỗi gửi cho AI ở đoạn chat mới, AI sẽ đọc và sửa ngay.

## Khi có thiết bị, chỉ cần nhắn AI:
- Loại máy: VPS thường (CPU) hay đã thuê kèm GPU (bao nhiêu VRAM)
- **RAM CPU** (lệnh `free -h`) và **loại ổ đĩa** (NVMe/SSD/ổ mạng) — 2 thứ này quyết định tốc độ ZeRO-Infinity, quan trọng hơn cả VRAM
- Dung lượng đĩa trống (lệnh `df -h`)

AI sẽ đưa đúng lệnh cần chạy dựa trên đó — không cần tự tính toán gì thêm.
