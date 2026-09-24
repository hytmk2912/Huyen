"""Runtime chạy việc cho agent, gộp từ repo Agent (xem docs/GOP_AGENT.md).

- `executor`: chạy một lệnh dạng list (không qua shell), có timeout, cắt bớt output, ghi job id và thời điểm.
- `jobs`: hàng đợi job trong bộ nhớ (tạo, xem, nhận, trả kết quả).
- `auth`: kiểm tra token kiểu "Bearer <token>" bằng so sánh an toàn.
Chỉ dùng thư viện chuẩn; không mở cổng mạng nào khi import.
"""
