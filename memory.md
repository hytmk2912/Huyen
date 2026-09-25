# Bộ nhớ làm việc

TUẦN 3

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng). Kế hoạch và tiêu chí: `TASKS.md` (tuần 3 ở đầu file). Nhật ký tuần 1–2: `archive/memory-tuan-1.md`, `archive/memory-tuan-2.md`.

## Tồn tuần 2
- Không có mốc tồn: M8–M14 đều xong, đã gộp vào `main` (PR #8 `12e86fc`, 206 test).
- Chưa kiểm chứng trên GPU hay model thật: QLoRA, 2 notebook Colab, agent với Ollama + `qwen3:4b` (chờ chủ repo chạy notebook).

## Checklist tuần 3
Chủ repo chọn từng mốc trong 7 đề xuất (cuối README); mốc nào được chọn thì mới có tiêu chí trong `TASKS.md`.
- [ ] M15 Số đo thật trên Colab (đã chọn 25/9)
- M16 Notebook bền hơn · M17 Model chính: LoRA chỉ phần ngôn ngữ, `dtype` · M18 Agent dùng model đã train (GGUF) · M19 Mở rộng eval · M20 Dữ liệu: gần trùng train/eval, thêm preset tiếng Việt · M21 Tổng kết tuần 3 (chưa chọn)

## Mốc đang làm
- M15 — bắt đầu 25/9, **dở: chờ số đo thật** (tiêu chí 3). Phần ghi số đo và lệnh `calibrate` đã xong.
  - Khi chủ repo chạy xong notebook: đọc `so_do/<model>.json` trong repo riêng tư `Hytmk2912/huyen-<model>-qlora` (connector Hugging Face đọc được), hoặc xem ảnh chụp bảng Bước 12;
  - sau đó sửa `GPUS["T4"]` trong `estimate.py`, sửa `TRAINING_FACTOR` (chỉ theo số đo của `light`), rồi thêm bảng số đo thật vào README.

## Việc chủ repo tự làm
1. Tạo token Hugging Face quyền **Write**, thêm vào Colab Secrets tên `HF_TOKEN` và bật Notebook access.
2. Chạy `notebooks/train_colab.ipynb` với `smoke` (khoảng 30 phút; hướng dẫn `docs/TRAIN_COLAB.md`).
3. Chạy lại với `light` (khoảng 2 giờ; Colab ngắt thì Run all lại).
4. Chạy `notebooks/agent_colab.ipynb`, gửi tỉ lệ thành công và trace của nhiệm vụ không đạt.
5. Archive repo Agent cũ `huyenytmk2912/agent`. Nếu `hytmk2912/Agent` có code mới hơn `78a3e25`, cấp quyền để gộp thêm.
6. Quyết định PR #4 (đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B).
7. Việc còn từ tuần 1: sửa mô tả repo trên GitHub ("Train Từ Số 0"); đổi mật khẩu máy chủ cũ còn trong lịch sử commit; cân nhắc chuyển repo sang Private; đọc lại giấy phép 3 dataset preset.

## Việc dở
- Nhánh làm việc tuần 3: `claude/nhiem-vu-tuan-jixv6i` (nhánh phiên được giao; dựng lại từ `main` tại `12e86fc` vì chỉ còn lịch sử đã gộp).

## Lỗi còn tồn
| Lỗi | Nơi |
| --- | --- |
| Chưa chạy thật trên GPU hay model thật: QLoRA, model ảnh + chữ, 2 notebook Colab, agent với Ollama + qwen3:4b. `target_modules: "all-linear"` có thể gắn LoRA vào phần xử lý ảnh. | `adapters.py`, `finetune.py`, `notebooks/` |
| transformers 5.17 cảnh báo `torch_dtype` đã cũ, nên đổi sang `dtype`. | `adapters.py`, `finetune.py` |

## Nhật ký tuần 3
| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 25/9 | Thiết lập tuần 3 + M15 (dở) | Chủ repo chọn M15. Dựng lại nhánh `claude/nhiem-vu-tuan-jixv6i` từ `main` (`12e86fc`); chuyển nhật ký tuần 2 sang `archive/memory-tuan-2.md`; thêm phần tuần 3 vào `TASKS.md`, cập nhật skill `lam-moc` và `CLAUDE.md`. Kiểm tra Hugging Face của chủ repo: chưa có repo `huyen-*-qlora`, tức là chưa chạy notebook, nên phần sửa hằng số bị chặn. Đã làm: `measurements.json` khi train, thời gian trong báo cáo eval/agent, lệnh `calibrate`, ô Bước 12 trong notebook. Lỗi gặp: khi chạy tiếp, TRL đếm lại token nhưng log cũ vẫn còn, nên `num_tokens` lấy nhầm số lần trước (đã lọc theo bước); hệ số VRAM từ model nhỏ vô nghĩa (đã đánh dấu). 215 test qua, compileall và secret-scan sạch. Tiếp theo: chờ số đo thật cho M15, hoặc mốc chủ repo chọn tiếp. |
