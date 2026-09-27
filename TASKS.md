# Nhiệm vụ tuần 4 (M23–M27)

Bắt đầu: 2026-09-27, từ `main` tại commit `db0f9d4` (338 test chạy qua, không test nào bị bỏ qua; compileall và secret-scan sạch). Nhánh làm việc: `claude/nhiem-vu-tuan-jixv6i`.

Kế hoạch và bằng chứng các tuần trước: tuần 1–2 (M1–M14) trong `archive/tasks-tuan-1-2.md`, tuần 3 (M15–M22) chép nguyên văn trong `archive/tasks-tuan-3.md`.

Cách làm như tuần 3: mỗi lượt **tối đa 1 mốc** bằng skill `lam-moc`; mốc chỉ **Xong** khi đạt mọi tiêu chí và `python -m local_ai.check` (không `--allow-skip`) xanh; khi đó tự gộp PR. Tiêu chí 5 mốc dưới đây do chủ repo đặt ngày 27/9; mỗi tiêu chí phải có test hoặc lệnh chứng minh.

## Bảng tiến độ tuần 4

| Mốc | Nội dung | Trạng thái | Tiến độ | Bằng chứng |
| --- | --- | --- | --- | --- |
| M23 | Gửi `max_tokens` cho model qua server | **Chưa làm** | 0/2 (0%) | — |
| M24 | Notebook Kaggle | **Chưa làm** | 0/5 (0%) | — |
| M25 | Model trung gian 14B abliterated | **Chưa làm** | 0/4 (0%) | — |
| M26 | Preset tiếng Việt thứ 2 | **Chưa làm** | 0/4 (0%) | — |
| M27 | Tổng kết tuần 4 | **Chưa làm** | 0/3 (0%) | — |
| **Tổng** | 5 mốc | 0 **Xong** | 0/5 mốc (0%) | `python -m local_ai.check` lúc bắt đầu: 338 test chạy qua (27/9) |

## M23: Gửi max_tokens
Mục tiêu: `max_new_tokens` hiện không giới hạn được độ dài câu trả lời của model qua server (Ollama, llama.cpp, vLLM), vì adapter không gửi `max_tokens` (phát hiện khi làm M18).
- [ ] 1. `openai_compatible` gửi `"max_tokens"` = `config.max_new_tokens` trong payload.
- [ ] 2. Test dùng server giả, kiểm tra payload nhận được có `max_tokens` đúng giá trị, cả giá trị mặc định 512 và giá trị tự đặt. `python -m local_ai.check` xanh.

## M24: Notebook Kaggle
- [ ] 1. `notebooks/train_kaggle.ipynb` sinh bằng `notebooks/build.py`.
- [ ] 2. Token đọc từ Kaggle Secrets; kết quả ghi ra `/kaggle/working`; train tiếp từ checkpoint trên HF Hub khi phiên bị ngắt.
- [ ] 3. Notebook hợp lệ theo nbformat, không kèm output, ghim phiên bản thư viện; mọi lệnh chạy được bằng `--dry-run`.
- [ ] 4. README có mục hướng dẫn chạy trên iPhone.
- [ ] 5. `python -m local_ai.check` xanh.

## M25: Model trung gian 14B abliterated
- [ ] 1. `configs/models/platform.json` thêm bản Huihui Qwen3 14B abliterated, ghim revision 40 ký tự lấy từ API Hugging Face (chỉ đọc metadata, không tải trọng số). Không có mạng thì dừng lại và ghi vào "Việc dở".
- [ ] 2. Thêm `configs/training/colab_14b.json` (QLoRA, fp16).
- [ ] 3. Test chứng minh `vram.py` ước tính vừa 16 GB với seq len đã chọn; nếu không vừa thì ghi seq len lớn nhất còn vừa.
- [ ] 4. `python -m local_ai.check` xanh.

## M26: Preset tiếng Việt thứ 2
- [ ] 1. Chỉ chọn dataset có giấy phép cho phép dùng.
- [ ] 2. Ghi giấy phép vào `docs/GIAY_PHEP_DATASET.md`.
- [ ] 3. Có fixture và test giống các preset cũ.
- [ ] 4. `python -m local_ai.check` xanh.

## M27: Tổng kết tuần 4
- [ ] 1. README, `docs/ARCHITECTURE.md`, `memory.md` khớp thực tế.
- [ ] 2. Có đề xuất tuần 5.
- [ ] 3. `python -m local_ai.check` xanh.
