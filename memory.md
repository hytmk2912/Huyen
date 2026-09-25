# Bộ nhớ làm việc

TUẦN 3

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng). Kế hoạch và tiêu chí: `TASKS.md` (tuần 3 ở đầu file). Nhật ký tuần 1–2: `archive/memory-tuan-1.md`, `archive/memory-tuan-2.md`.

## Tồn tuần 2
- Không có mốc tồn: M8–M14 đều xong, đã gộp vào `main` (PR #8 `12e86fc`, 206 test).
- Chưa kiểm chứng trên GPU hay model thật: QLoRA, 2 notebook Colab, agent với Ollama + `qwen3:4b` (chờ chủ repo chạy notebook).

## Checklist tuần 3
Chủ repo chọn từng mốc trong 7 đề xuất (cuối README); mốc nào được chọn thì mới có tiêu chí trong `TASKS.md`.
- [ ] M15 Số đo thật trên Colab (đã chọn 25/9; **bị chặn**: chờ số đo thật)
- [x] M16 Notebook bền hơn (xong 25/9)
- M17 Model chính: LoRA chỉ phần ngôn ngữ (phần `dtype` đã làm khi rà soát 25/9) · M18 Agent dùng model đã train (GGUF) · M19 Mở rộng eval · M20 Dữ liệu: gần trùng train/eval, thêm preset tiếng Việt · M21 Tổng kết tuần 3 (chưa chọn)

## Mốc đang làm
- Không có mốc dở. M15 vẫn chờ số đo thật; mốc kế tiếp: mốc chủ repo chọn (M17–M21).
- M15 **bị chặn** 3 lượt liền (25/9): Hugging Face của chủ repo chỉ có `personal-ai-hytmk` (18/9), chưa có repo `huyen-*-qlora`, nên chưa có số đo thật (tiêu chí 3). PR #9 đã gộp (`4eaf1e9`); cần chủ repo chạy notebook train; không đoán số. Phần ghi số đo và lệnh `calibrate` đã xong.
  - Khi chủ repo chạy xong notebook: đọc `so_do/<model>.json` trong repo riêng tư `Hytmk2912/huyen-<model>-qlora` (connector Hugging Face đọc được), hoặc xem ảnh chụp bảng Bước 12;
  - sau đó sửa `GPUS["T4"]` trong `estimate.py`, sửa `TRAINING_FACTOR` (chỉ theo số đo của `light`), rồi thêm bảng số đo thật vào README.

## Việc chủ repo tự làm (rà soát M1–M16 ngày 25/9)
1. Tạo token Hugging Face quyền **Write**, thêm vào Colab Secrets tên `HF_TOKEN` và bật Notebook access.
2. Chạy `notebooks/train_colab.ipynb` với `smoke` (khoảng 30 phút; hướng dẫn `docs/TRAIN_COLAB.md`), rồi với `light` (khoảng 2 giờ; Colab ngắt thì Run all lại). Xong thì gọi M15.
3. Chạy `notebooks/agent_colab.ipynb`, gửi tỉ lệ thành công và trace của nhiệm vụ không đạt.
4. Đổi mật khẩu rsync (user `huyen`) đã lộ trong lịch sử commit `e6bc723` (file `cpu_hub.sh`, `gpu_8h.sh`, đã xóa nhưng lịch sử vẫn công khai). Không chép mật khẩu vào đâu cả.
5. Quyết định PR #4: đề nghị đóng (xung đột khoảng 65 file, xóa agent; phần sửa dữ liệu đã có trong `main`).
6. Xóa nhánh `codex/build-autonomous-ai-system-architecture` (đã gộp hết vào `main`; Claude không có quyền xóa nhánh).
7. Archive repo Agent cũ `huyenytmk2912/agent` (không đổi từ `78a3e25`). `hytmk2912/Agent`: Claude không truy cập được; chỉ cấp quyền nếu repo đó có code mới hơn.
8. Sửa mô tả repo trên GitHub (trang chính repo → mục About → biểu tượng ⚙️), hiện vẫn là "Train Từ Số 0"; cân nhắc chuyển repo sang Private (khi đó `git clone` trong notebook phải dùng token).
9. Đọc `docs/GIAY_PHEP_DATASET.md`, chấp nhận hoặc bỏ preset (`vietnamese` chỉ dùng cá nhân, không thương mại).
10. Model chính 27B cần GPU 24 GB trở lên (Colab free không đủ). Chọn mốc tiếp theo trong M17–M21.

## Việc dở
- Nhánh làm việc tuần 3: `claude/nhiem-vu-tuan-jixv6i` (nhánh phiên được giao; dựng lại từ `main` tại `4eaf1e9` sau khi gộp PR #9).

## Lỗi còn tồn
| Lỗi | Nơi |
| --- | --- |
| Chưa chạy thật trên GPU hay model thật: QLoRA, model ảnh + chữ, 2 notebook Colab, agent với Ollama + qwen3:4b. `target_modules: "all-linear"` có thể gắn LoRA vào phần xử lý ảnh. | `adapters.py`, `finetune.py`, `notebooks/` |

## Nhật ký tuần 3
| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 25/9 | Thiết lập tuần 3 + M15 (dở) | Chủ repo chọn M15. Dựng lại nhánh `claude/nhiem-vu-tuan-jixv6i` từ `main` (`12e86fc`); chuyển nhật ký tuần 2 sang `archive/memory-tuan-2.md`; thêm phần tuần 3 vào `TASKS.md`, cập nhật skill `lam-moc` và `CLAUDE.md`. Kiểm tra Hugging Face của chủ repo: chưa có repo `huyen-*-qlora`, tức là chưa chạy notebook, nên phần sửa hằng số bị chặn. Đã làm: `measurements.json` khi train, thời gian trong báo cáo eval/agent, lệnh `calibrate`, ô Bước 12 trong notebook. Lỗi gặp: khi chạy tiếp, TRL đếm lại token nhưng log cũ vẫn còn, nên `num_tokens` lấy nhầm số lần trước (đã lọc theo bước); hệ số VRAM từ model nhỏ vô nghĩa (đã đánh dấu). 215 test qua, compileall và secret-scan sạch. Tiếp theo: chờ số đo thật cho M15, hoặc mốc chủ repo chọn tiếp. |
| 25/9 | M16 | Xong 4/4. Hàm `run` (`local_ai/colab.py`): lệnh chạy không qua shell, in output ngay khi có, lỗi thì ném `StepFailed` nên Run all dừng; 2 notebook bỏ hết `!python`/`!pip`/`!apt-get`/`!ollama` (chỉ còn `!git clone` lần đầu). Lệnh eval có `--hub-repo`/`--hub-path`: đẩy báo cáo kèm cài đặt, chạy lại thì dùng lại nếu cài đặt khớp (Bước 7 không chấm lại sau khi Colab ngắt). Không dùng `_exit_code` của Colab vì không kiểm chứng được (không có `google-colab` trên PyPI). Lỗi gặp: báo cáo hỏng trên Hub làm lệnh chấm lỗi, nay chấm lại; một test dùng chung thư mục Hub giả nên sai, đã tách. Test M10/M13 sửa cách đọc lệnh. M15 vẫn bị chặn (lượt 2). 226 test qua, compileall và secret-scan sạch. |
| 25/9 | M15 (kiểm tra lại) | Chủ repo gọi M15. Hugging Face vẫn chưa có repo `huyen-*-qlora` (chỉ có `personal-ai-hytmk` từ 18/9), PR #9 chưa gộp: tiêu chí 3 vẫn bị chặn, không sửa hằng số. Không đổi code. |
| 25/9 | Rà soát M1–M16 | Chủ repo nhờ rà soát việc cần chủ repo làm, việc làm được thì làm luôn. PR #9 đã gộp (`4eaf1e9`); dựng lại nhánh từ `main`. Đã làm: nạp model dùng `dtype` thay `torch_dtype` (hết cảnh báo transformers 5.x); secret-scan bắt kiểu mật khẩu shell từng lộ (`BIẾN_PASS="${BIẾN_PASS:-...}"`); `.gitignore` thêm `*.bin`, `.ruff_cache/` (lấy từ PR #4); `docs/GIAY_PHEP_DATASET.md` (`vietnamese` gốc CC BY-NC 4.0: không thương mại); README cập nhật trạng thái M15/M16. Kiểm tra: nhánh `codex/...` đã gộp hết (xóa bị từ chối quyền, chuyển cho chủ repo); PR #4 nên đóng; `hytmk2912/Agent` không truy cập được; Hugging Face chưa có số đo. Test mới `tests/test_ra_soat_m1_m16.py` (6 test). |
