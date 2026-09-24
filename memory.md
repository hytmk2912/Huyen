# Các quyết định đã chốt

Ghi lại để các phiên sau (người hoặc AI) không làm ngược. Muốn đổi quyết định nào thì hỏi chủ repo và cập nhật file này.

| Ngày | Quyết định | Lý do |
| --- | --- | --- |
| 2026-09-24 | Phạm vi repo chỉ là fine-tune: dataset HF → sft.jsonl → LoRA/QLoRA → đánh giá. | Mỗi phiên AI trước đây tự thêm tính năng (agent, corpus 10T) làm repo rối. |
| 2026-09-24 | Bỏ phần corpus 10T token (pretrain). | Quy mô pretrain (Llama 3 dùng khoảng 15T), không liên quan fine-tune. Code cũ còn trong lịch sử git. |
| 2026-09-24 | Bỏ agent, tools, memory, router, demo khỏi repo này; sẽ chuyển sang repo Agent riêng. | Ngoài phạm vi fine-tune. Bản đầy đủ cuối cùng nằm ở commit `2626c7d`. |
| 2026-09-24 | Bỏ `archive/legacy-nanogpt/`. | Hướng dẫn cho các script đã bị xóa. |
| 2026-09-24 | Giữ model ở bf16, không đổi lên FP32/GGUF f32. | Trọng số gốc là bf16; FP32 không làm model tốt hơn, chỉ tốn gấp đôi bộ nhớ (bản 14B khoảng 56 GB). |
| 2026-09-24 | Model phụ: Huihui Qwen3 4B/8B/14B abliterated-v2. | Chủ repo chọn dòng Huihui Qwen3; Qwen3 không có cỡ 3B/7B nên dùng 4B/8B. |
| 2026-09-24 | Thêm QLoRA (4-bit NF4) cho model chính 27B. | bf16 cần khoảng 54 GB chỉ cho trọng số (GPU 80 GB); QLoRA khoảng 15 GB, chạy được trên GPU 24 GB. |
| 2026-09-24 | sft.jsonl giữ nguyên hội thoại nhiều lượt, giữ reasoning (khối `<think>`) và context. | Trước đây ghép sai lượt và làm mất phần suy luận của Qwen3. |
| 2026-09-24 | Chặn rò rỉ eval bằng ID + content_hash + câu hỏi đã chuẩn hóa; chỉ loại câu trùng, không dừng chương trình. | So riêng ID thì câu y hệt khác ID vẫn lọt. |
| 2026-09-24 | Giữ lại từ nhánh cũ chỉ phần đánh giá sau khi train (`evaluation/suites.py`, `data/eval/vi_trading_eval.jsonl`). | Phần còn lại của nhánh là mở rộng corpus/agent ngoài phạm vi. |
| 2026-09-24 | Bỏ quy tắc "lộ trình luôn 5 mục" và vòng hẹn giờ 30 phút. | Đó là cơ chế tự thêm tính năng không ngừng. |
