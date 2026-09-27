# Bộ nhớ làm việc

TUẦN 3

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. **Tối đa 5.000 ký tự.** Kế hoạch: `TASKS.md` (tuần 3); tuần 1–2: `archive/tasks-tuan-1-2.md`. Nhật ký chi tiết: `archive/memory-tuan-1.md`, `archive/memory-tuan-2.md`, `archive/memory-tuan-3.md`.

## Tồn tuần trước
- Không có: M1–M14 đều xong. Từ M15 đã chạy thật trên Colab T4 (25–26/9): QLoRA `smoke`, `light`, 2 notebook, agent Ollama + `qwen3:4b`.

## Checklist tuần 3
- [x] M15 Số đo thật trên Colab · [x] M16 Notebook bền hơn · [x] M17 LoRA chỉ phần ngôn ngữ
- [x] M19 Chấm công bằng hơn · [x] M20 Dữ liệu giữ khả năng gọi công cụ · [x] M22 Kiểm định ý nghĩa cho so sánh eval
- [x] M18 Agent dùng model đã train (GGUF) · [x] M21 Tổng kết tuần 3

## Mốc đang làm
- **XONG TUẦN 3** (8/8 mốc đã chọn, 100%: M15–M22; M21 xong 27/9). Mọi phần đã code và test; phần M18–M20 chưa chạy lại trên Colab (việc chủ repo 8, 9). Chờ chủ repo chọn việc tuần 4, không tự làm.
- Tự gộp PR khi `python -m local_ai.check` (không `--allow-skip`) xanh; không hỏi chủ repo (chủ repo cho phép 26/9).
- Số đo thật (M15): `tests/fixtures/measurements/that_*.json`; T4 `train_tflops` 5,2; VRAM QLoRA cộng `KBIT_OVERHEAD_GB` × √(tỷ tham số); agent 4/5.

## Đề xuất tuần 4
- Chạy lại `train_colab` (smoke, rồi light) để có điểm mới sau M19, M20; rồi chạy `agent_trained_colab` (M18) để so agent trước/sau khi train.
- Adapter server gửi `max_tokens` = `max_new_tokens`, để giới hạn độ dài câu trả lời có tác dụng với model qua Ollama (phát hiện khi làm M18).
- Thêm một preset tiếng Việt nữa, sau khi đọc kỹ giấy phép.
- Train thật model chính 27B trên GPU 40–48 GB (chủ repo thuê GPU).

## Việc chủ repo tự làm
1. Đổi mật khẩu rsync (user `huyen`) lộ trong lịch sử commit `e6bc723`. Không chép mật khẩu vào đâu.
2. Xoá nhánh `claude/expand-model-training-repo-v2yzyr` (PR #4 đã đóng 27/9; Claude không có quyền xoá).
3. Xóa nhánh `codex/build-autonomous-ai-system-architecture` (đã gộp hết vào `main`; Claude không có quyền xóa).
4. Archive repo `huyenytmk2912/agent`; `hytmk2912/Agent` chỉ cấp quyền nếu có code mới hơn.
5. Sửa mô tả repo (trang repo → About → ⚙️), hiện là "Train Từ Số 0"; cân nhắc chuyển Private (khi đó `git clone` trong notebook cần token).
6. Đọc `docs/GIAY_PHEP_DATASET.md` (`vietnamese` chỉ dùng cá nhân, không thương mại).
7. Model chính 27B: cần GPU 40–48 GB để train thật; thuê GPU rồi báo.
8. Chạy lại `train_colab` (smoke, rồi light) để có điểm mới sau M19, M20: 38 câu, greedy, tắt suy nghĩ, 10% dòng gọi công cụ, loss chỉ trên câu trả lời. Gửi ảnh Bước 10 và 12.
9. Sau mục 8: chạy `agent_trained_colab` (M18) cùng model; gửi ảnh Bước 11 (bảng so sánh model gốc và model đã train).

## Việc dở
- Nhánh: `claude/nhiem-vu-tuan-jixv6i`, dựng lại từ `main` sau mỗi lần gộp PR.
- Từ M19 bộ chấm 38 câu, greedy, notebook tắt suy nghĩ: điểm cũ (30 câu, lấy mẫu) không so trực tiếp với điểm mới. Revision model đã ghim (26/9); muốn dùng bản mới của repo model thì sửa `platform.json`.
- Việc gửi `max_tokens` (phát hiện ở M18) đã đưa vào đề xuất tuần 4.

## Lỗi còn tồn
| Lỗi | Nơi |
| --- | --- |
| Model chính 27B chưa train thật trên GPU; QLoRA 4bit với model ảnh + chữ chưa chạy trên GPU. | `finetune.py`, `configs/training/qlora_primary.json` |
| Sau khi train, tool_use của `light` tụt 7/8 → 3/8 (đo 26/9; p = 0,125 theo M22, chưa chắc là tụt thật). M19, M20 đã sửa bộ chấm và dữ liệu train; chưa chạy lại trên Colab để biết đã giữ được chưa. | Colab (chủ repo) |

## Nhật ký tuần 3
Mỗi lượt một dòng ngắn; chi tiết trong `archive/memory-tuan-3.md`.

| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 25–26/9 | M15 | Số đo thật `smoke`, `light`, agent; sửa ước tính thời gian và VRAM; xong 26/9 (PR #11–#14). |
| 26/9 | M20 | `--tool-calls 0.1` (200/2000 dòng gọi công cụ tự sinh, notebook bật), chặn gần trùng với bộ chấm bằng MinHash (`eval_near_duplicate` trong manifest), `assistant_only_loss` (template Qwen thật TRL thay được; không hỗ trợ thì báo lỗi trước khi nạp model). 309 test, check xanh. Xong 5/5 mốc đã chọn. |
| 27/9 | Quy ước làm việc (chủ repo yêu cầu, không phải mốc) | `CLAUDE.md` thêm "5 nguyên tắc làm việc", dừng hỏi, chống quên khi nén ngữ cảnh, chia việc cho subagent; skill `lam-moc` thêm 3 dòng. 309 test, check xanh. PR #20 gộp; PR #4 đóng, chưa xoá được nhánh (403). |
| 27/9 | M22 | `compare` in p-value McNemar chính xác (chỉ `math.comb`) cho tổng, nhóm, ngôn ngữ; `--output` ghi `report.md`. tool_use 7/8 → 3/8: p = 0,125. 317 test, check xanh. Xong 6/6 mốc đã chọn. |
| 27/9 | M18 | Lệnh `export` (gộp adapter, GGUF bằng llama.cpp `b11205`, Modelfile theo template Ollama), `pull-adapter`, `agents.compare`, notebook `agent_trained_colab`. Thử thật trên máy phát triển với llama.cpp và Ollama 0.34.4 (model tí hon). 333 test (sau khi gộp M22), check xanh. Xong 7/7 mốc đã chọn. |
| 27/9 | M21 | Tổng kết tuần 3: README (trạng thái, lộ trình 8 mốc, đề xuất tuần 4, PR #4 đã đóng), ARCHITECTURE (phần chưa kiểm chứng), memory. Xong 8/8 mốc. |
