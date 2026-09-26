# Bộ nhớ làm việc

TUẦN 3

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. **Tối đa 5.000 ký tự.** Kế hoạch: `TASKS.md` (tuần 3); tuần 1–2: `archive/tasks-tuan-1-2.md`. Nhật ký chi tiết: `archive/memory-tuan-1.md`, `archive/memory-tuan-2.md`, `archive/memory-tuan-3.md`.

## Tồn tuần trước
- Không có: M1–M14 đều xong. Từ M15 đã chạy thật trên Colab T4 (25–26/9): QLoRA `smoke`, `light`, 2 notebook, agent Ollama + `qwen3:4b`.

## Checklist tuần 3
- [x] M15 Số đo thật trên Colab · [x] M16 Notebook bền hơn · [x] M17 LoRA chỉ phần ngôn ngữ
- [x] M19 Chấm công bằng hơn · [x] M20 Dữ liệu giữ khả năng gọi công cụ
- Chưa chọn: M18 Agent dùng model đã train (GGUF) · M21 Tổng kết tuần 3

## Mốc đang làm
- **XONG TUẦN 3** (5/5 mốc đã chọn: M15, M16, M17, M19, M20; M20 xong 26/9). M18, M21 chưa có tiêu chí trong `TASKS.md`: chờ chủ repo chọn, không tự làm.
- Tự gộp PR khi `python -m local_ai.check` (không `--allow-skip`) xanh; không hỏi chủ repo (chủ repo cho phép 26/9).
- Số đo thật (M15): `tests/fixtures/measurements/that_*.json`; T4 `train_tflops` 5,2; VRAM QLoRA cộng `KBIT_OVERHEAD_GB` × √(tỷ tham số); agent 4/5.

## Việc chủ repo tự làm
1. Đổi mật khẩu rsync (user `huyen`) lộ trong lịch sử commit `e6bc723`. Không chép mật khẩu vào đâu.
2. Đóng PR #4 (đề nghị đóng: xung đột khoảng 65 file, xóa agent; phần sửa dữ liệu đã có trong `main`).
3. Xóa nhánh `codex/build-autonomous-ai-system-architecture` (đã gộp hết vào `main`; Claude không có quyền xóa).
4. Archive repo `huyenytmk2912/agent`; `hytmk2912/Agent` chỉ cấp quyền nếu có code mới hơn.
5. Sửa mô tả repo (trang repo → About → ⚙️), hiện là "Train Từ Số 0"; cân nhắc chuyển Private (khi đó `git clone` trong notebook cần token).
6. Đọc `docs/GIAY_PHEP_DATASET.md` (`vietnamese` chỉ dùng cá nhân, không thương mại).
7. Model chính 27B: cần GPU 40–48 GB để train thật; thuê GPU rồi báo.
8. Chạy lại `train_colab` (smoke, rồi light) để có điểm mới sau M19, M20: 38 câu, greedy, tắt suy nghĩ, 10% dòng gọi công cụ, loss chỉ trên câu trả lời. Gửi ảnh Bước 10 và 12.

## Việc dở
- Nhánh: `claude/nhiem-vu-tuan-jixv6i`, dựng lại từ `main` sau mỗi lần gộp PR.
- Từ M19 bộ chấm 38 câu, greedy, notebook tắt suy nghĩ: điểm cũ (30 câu, lấy mẫu) không so trực tiếp với điểm mới. Revision model đã ghim (26/9); muốn dùng bản mới của repo model thì sửa `platform.json`.

## Lỗi còn tồn
| Lỗi | Nơi |
| --- | --- |
| Model chính 27B chưa train thật trên GPU; QLoRA 4bit với model ảnh + chữ chưa chạy trên GPU. | `finetune.py`, `configs/training/qlora_primary.json` |
| Sau khi train, tool_use của `light` tụt 7/8 → 3/8 (đo 26/9). M19, M20 đã sửa bộ chấm và dữ liệu train; chưa chạy lại trên Colab để biết đã giữ được chưa. | Colab (chủ repo) |

## Nhật ký tuần 3
Mỗi lượt một dòng ngắn; chi tiết trong `archive/memory-tuan-3.md`.

| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 25/9 | M15 (phần ghi số đo), M16 | M16 xong; M15 chờ số đo thật. |
| 25/9 | Rà soát M1–M16; sửa lỗi Bước 8 (fp16 trên T4) | PR #10 đã gộp. |
| 25–26/9 | M15 | Số đo thật `smoke`, `light`, agent; sửa ước tính thời gian và VRAM; xong 26/9 (PR #11–#14). |
| 26/9 | M17 | LoRA chỉ gắn phần ngôn ngữ của model ảnh + chữ; xong (PR #15). |
| 26/9 | Rà soát tuần 3 | README khớp thực tế; tuần 1–2 của `TASKS.md` sang `archive/tasks-tuan-1-2.md`; `memory.md` ≤ 5.000 ký tự; lệnh `python -m local_ai.check`; `run` đóng pipe; ghi tiêu chí M19, M20. 266 test, check xanh. Tiếp theo: M19. |
| 26/9 | Sửa 2 lỗi bảo mật | TerminalTool: git chỉ thấy repo trong thư mục làm việc (`GIT_CEILING_DIRECTORIES`); sandbox và TerminalTool dùng môi trường tối thiểu, không lộ `HF_TOKEN`. Ghi thêm tiêu chí M19 (8) và M20 (4). 271 test, check xanh. Tiếp theo: M19. |
| 26/9 | M19 | Tắt suy nghĩ (`--no-thinking`), chấm sau đẩy `eval/sau` (`--no-reuse`), 16 câu tool_use (38 câu), gợi ý khi TerminalTool từ chối + nhiệm vụ thử lại, chấm greedy lặp lại được, so số theo giá trị, ghim revision. 294 test, check xanh. Tiếp theo: M20. |
| 26/9 | M20 | `--tool-calls 0.1` (200/2000 dòng gọi công cụ tự sinh, notebook bật), chặn gần trùng với bộ chấm bằng MinHash (`eval_near_duplicate` trong manifest), `assistant_only_loss` (template Qwen thật TRL thay được; không hỗ trợ thì báo lỗi trước khi nạp model). 309 test, check xanh. Xong 5/5 mốc đã chọn. |
