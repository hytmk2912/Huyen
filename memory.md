# Bộ nhớ làm việc

TUẦN 4

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. **Tối đa 5.000 ký tự.** Kế hoạch: `TASKS.md` (tuần 4); tuần 1–2: `archive/tasks-tuan-1-2.md`; tuần 3: `archive/tasks-tuan-3.md`. Nhật ký chi tiết: `archive/memory-tuan-1.md`, `archive/memory-tuan-2.md`, `archive/memory-tuan-3.md` (có bản cuối memory tuần 3), `archive/memory-tuan-4.md`.

## Tồn tuần trước
- Không có mốc tồn: tuần 3 xong 8/8 mốc (M15–M22). Phần M18–M20 chưa chạy lại trên Colab (việc chủ repo 8, 9).

## Checklist tuần 4
- [x] M23 Gửi max_tokens · [x] M24 Notebook Kaggle · [ ] M25 Model 14B abliterated
- [ ] M26 Preset tiếng Việt thứ 2 · [ ] M27 Tổng kết tuần 4

## Mốc đang làm
- M24 xong 27/9. Mốc kế tiếp: **M25** (Model 14B abliterated), chờ chủ repo gọi.
- Tự gộp PR khi `python -m local_ai.check` (không `--allow-skip`) xanh; không hỏi chủ repo (chủ repo cho phép 26/9).
- Số đo thật (M15): `tests/fixtures/measurements/that_*.json`; T4 `train_tflops` 5,2; VRAM QLoRA cộng `KBIT_OVERHEAD_GB` × √(tỷ tham số); agent 4/5.

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
- `train_kaggle` (M24) chưa chạy trên Kaggle thật; cách chạy trên iPhone ở README mục "Train trên Kaggle miễn phí". Chạy rồi thì gửi ảnh Bước 10 và 12.
- Từ M19 bộ chấm 38 câu, greedy, notebook tắt suy nghĩ: điểm cũ (30 câu, lấy mẫu) không so trực tiếp với điểm mới. Revision model đã ghim (26/9).

## Lỗi còn tồn
| Lỗi | Nơi |
| --- | --- |
| Model chính 27B chưa train thật trên GPU; QLoRA 4bit với model ảnh + chữ chưa chạy trên GPU. | `finetune.py`, `configs/training/qlora_primary.json` |
| Sau khi train, tool_use của `light` tụt 7/8 → 3/8 (đo 26/9; p = 0,125 theo M22, chưa chắc là tụt thật). M19, M20 đã sửa bộ chấm và dữ liệu train; chưa chạy lại trên Colab. | Colab (chủ repo) |

## Nhật ký tuần 4
Mỗi lượt một dòng ngắn; chi tiết trong `archive/memory-tuan-4.md`.

| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 27/9 | Mở tuần 4 | `TASKS.md` tuần 3 chép nguyên văn sang `archive/tasks-tuan-3.md`; `TASKS.md` mới M23–M27; nhật ký tuần 3 sang `archive/memory-tuan-3.md`. |
| 27/9 | M23 | Adapter server gửi `max_tokens` = `max_new_tokens`; 4 test mới với server giả (mặc định 512, tự đặt, `--max-new-tokens` của lệnh chấm). 342 test, KẾT QUẢ: XANH. Tiếp theo: M24. |
| 27/9 | M24 | Notebook `train_kaggle` (Kaggle Secrets, kết quả trong `/kaggle/working`, train tiếp từ checkpoint trên Hub, chung repo với Colab); README hướng dẫn iPhone; `calibrate --hub-path`. 17 test mới; 359 test, KẾT QUẢ: XANH. Tiếp theo: M25. |
