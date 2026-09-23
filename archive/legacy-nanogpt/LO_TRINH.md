# Lộ Trình Dự Án — Theo Giai Đoạn

**Cách đọc**: mỗi giai đoạn có % hoàn thành thật (không phải ước lượng lạc quan) và thời gian ước tính. Giai đoạn 2 là giai đoạn KHÔNG đoán trước chính xác được — phụ thuộc bạn đầu tư bao nhiêu thời gian/tiền GPU.

## Giai đoạn 0 — Lên kế hoạch, viết script
**100% xong.** Chọn hướng (train từ số 0, không dùng model có sẵn), viết toàn bộ script trong repo này, sửa qua nhiều vòng review kỹ thuật thật.

## Giai đoạn 1 — Chuẩn bị dữ liệu
**~95% xong.** Đã chạy Bước 1 (~14 tỷ token) + Bước 1b (thêm từ FineWeb-Edu 100BT, lấp đầy đĩa còn trống). Còn thiếu: xác nhận Bước 1b chạy xong hoàn toàn (chờ thông báo ntfy).
**Thời gian**: vài giờ đến 1 ngày — gần như xong.

## Giai đoạn 2 — Train model (~2,7B tham số)
**0% xong — chưa bắt đầu, chưa thuê GPU.**
**Thời gian**: đây là phần KHÔNG đoán chính xác được. Model cần rất nhiều lượt train qua dữ liệu mới học tốt (đã nói ở các tin trước: dữ liệu hiện có vẫn ít hơn mức lý tưởng). Có thể là:
- Vài phiên GPU (vài chục giờ tổng) → có model chạy được nhưng còn yếu.
- Nhiều chục phiên trải dài vài tuần đến vài tháng → model khá hơn rõ rệt.
- Không có mốc "xong hẳn" cứng — càng train nhiều càng tốt hơn, dừng lúc nào là do bạn quyết định (đủ tốt để dùng, hoặc hết ngân sách/thời gian muốn đầu tư).
**Việc cụ thể**: thuê GPU → chạy thử ngắn (bắt buộc) → train → tắt máy → lặp lại, script tự tiếp tục từ checkpoint mỗi lần (đã sửa ở các tin trước).

## Giai đoạn 3 — Đánh giá kết quả
**0% xong — cần có checkpoint từ Giai đoạn 2 trước.**
**Thời gian**: nhanh (vài phút mỗi lần chạy `sample_3b.py`), làm xen kẽ suốt Giai đoạn 2 để biết model tiến bộ tới đâu, không phải đợi Giai đoạn 2 "xong" mới làm.

## Giai đoạn 4 — Fine-tune/Alignment (Bước 2 trong khung "2 bước" đã thống nhất)
**0% xong — cố tình chưa làm, đợi Giai đoạn 2 có kết quả đủ tốt.**
**Thời gian**: thường NHẸ hơn Giai đoạn 2 nhiều (fine-tune cần ít dữ liệu/thời gian train hơn pretrain) — có thể vài phiên GPU là đủ, một khi tới lúc làm.

## Giai đoạn 5 — Lớp Agent (tự lên kế hoạch, gọi công cụ, tự sửa lỗi)
**0% xong — đây là mục tiêu ban đầu của cả dự án, nhưng là lớp NGOÀI CÙNG, làm sau khi có model dùng được.**
**Thời gian**: chủ yếu là việc lập trình (nối model với vòng lặp tự động, giống hướng `loop.sh` đã bàn ở giai đoạn đầu dự án) — không cần train thêm nhiều, có thể làm nhanh (vài ngày) một khi tới lượt.

---
**Tóm lại đang ở đâu**: vừa xong Giai đoạn 1, sắp bước vào Giai đoạn 2 — giai đoạn dài và không chắc thời gian nhất trong cả 5 giai đoạn.
