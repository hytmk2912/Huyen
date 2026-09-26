# Bộ nhớ làm việc

TUẦN 3

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng). Kế hoạch và tiêu chí: `TASKS.md` (tuần 3 ở đầu file). Nhật ký tuần 1–2: `archive/memory-tuan-1.md`, `archive/memory-tuan-2.md`.

## Tồn tuần 2
- Không có mốc tồn: M8–M14 đều xong, đã gộp vào `main` (PR #8 `12e86fc`, 206 test).
- Đã kiểm chứng trên Colab T4 (25–26/9, M15): QLoRA `smoke` và `light`, 2 notebook Colab, agent với Ollama + `qwen3:4b`.

## Checklist tuần 3
Chủ repo chọn từng mốc trong 7 đề xuất (cuối README); mốc nào được chọn thì mới có tiêu chí trong `TASKS.md`.
- [x] M15 Số đo thật trên Colab (xong 26/9)
- [x] M16 Notebook bền hơn (xong 25/9)
- M17 Model chính: LoRA chỉ phần ngôn ngữ (phần `dtype` đã làm khi rà soát 25/9) · M18 Agent dùng model đã train (GGUF) · M19 Mở rộng eval · M20 Dữ liệu: gần trùng train/eval, thêm preset tiếng Việt · M21 Tổng kết tuần 3 (chưa chọn)

## Mốc đang làm
- Không có mốc dở. M15 xong 26/9 (4/4); mốc kế tiếp: mốc chủ repo chọn trong M17–M21.
- Tóm tắt M15 (chi tiết trong `TASKS.md`): số đo thật trên T4 của `smoke`, `light`, agent chép ở `tests/fixtures/measurements/that_*.json` (số đo train của `light` trên Hub đã bị lần chạy 0 bước ghi đè). Đã sửa `train_tflops` T4 = 5,2; VRAM QLoRA thêm `KBIT_OVERHEAD_GB` × √(tỷ tham số) (27B ước tính 34,6 GB, chưa đo); `token_overhead_s` giữ nguyên có căn cứ. Agent 4/5 (80%), 11,1 phút.

## Việc chủ repo tự làm (rà soát M1–M16 ngày 25/9)
1. ~~Tạo token Hugging Face, chạy `train_colab` (`smoke`, `light`) và `agent_colab`~~ (xong 25–26/9).
2. Chọn mốc tiếp theo trong M17–M21; chọn cách giữ tool_use khi train (README, hợp M19/M20); có tắt chế độ suy nghĩ khi chấm Qwen3 không.
3. Model chính 27B: ước tính khoảng 35 GB khi train QLoRA, cần GPU 40–48 GB (Colab free không đủ; đo ở M17).
4. Đổi mật khẩu rsync (user `huyen`) đã lộ trong lịch sử commit `e6bc723` (file `cpu_hub.sh`, `gpu_8h.sh`, đã xóa nhưng lịch sử vẫn công khai). Không chép mật khẩu vào đâu cả.
5. Quyết định PR #4: đề nghị đóng (xung đột khoảng 65 file, xóa agent; phần sửa dữ liệu đã có trong `main`).
6. Xóa nhánh `codex/build-autonomous-ai-system-architecture` (đã gộp hết vào `main`; Claude không có quyền xóa nhánh).
7. Archive repo Agent cũ `huyenytmk2912/agent` (không đổi từ `78a3e25`). `hytmk2912/Agent`: Claude không truy cập được; chỉ cấp quyền nếu repo đó có code mới hơn.
8. Sửa mô tả repo trên GitHub (trang chính repo → mục About → biểu tượng ⚙️), hiện vẫn là "Train Từ Số 0"; cân nhắc chuyển repo sang Private (khi đó `git clone` trong notebook phải dùng token).
9. Đọc `docs/GIAY_PHEP_DATASET.md`, chấp nhận hoặc bỏ preset (`vietnamese` chỉ dùng cá nhân, không thương mại).

## Việc dở
- Nhánh làm việc tuần 3: `claude/nhiem-vu-tuan-jixv6i` (nhánh phiên được giao; dựng lại từ `main` tại `c7f4cc7` sau khi gộp PR #13).
- Agent thật (26/9): khi TerminalTool từ chối lệnh (ví dụ awk có `>`), `qwen3:4b` không thử lại bằng lệnh khác mà đoán. Có thể thêm gợi ý lệnh được phép vào thông báo từ chối; hợp M18/M19, hỏi chủ repo trước.
- Chấm Qwen3 (`light`): model "suy nghĩ" trong `<think>` hết 512 token trước khi viết code, nên code chỉ đạt 1/8 (lỗi NameError). Cách sửa có thể: tắt chế độ suy nghĩ khi chấm (`enable_thinking=False`) hoặc tăng `max_new_tokens`. Việc ngoài M15 (hợp với M19), hỏi chủ repo trước.
- Giữ tool_use khi train (`light` 7/8 → 3/8 sau train): đề xuất 4 cách trong README ("Vì sao tool_use tụt"); nên làm trước: trộn khoảng 10% dữ liệu gọi công cụ tự sinh (hợp M20). Chờ chủ repo chọn mốc.
- Điểm `smoke` sau khi train giảm 16 → 14/30, nhưng báo cáo chấm sau (Bước 9) không được đẩy lên Hub nên chưa xem được câu sai thêm. Muốn đẩy thì phải làm cách khác Bước 7 (Bước 9 luôn chấm lại): việc ngoài M15, hỏi chủ repo trước.

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
| 25/9 | Sửa lỗi Bước 8 trên T4 | Chủ repo chạy `smoke` trên T4: Bước 8 lỗi `_amp_foreach_non_finite_check_and_unscale_cuda ... 'BFloat16'`. Nguyên nhân: TRL 1.13 đổi tham số LoRA của model nạp 4bit sang bf16 ngay trong `SFTTrainer(...)`; train fp16 thì GradScaler không nhận gradient bf16. transformers 5.17 vẫn đọc `torch_dtype` (chỉ cảnh báo), nên không phải do `torch_dtype`. Sửa: `trainable_to_float32` đổi tham số được train sang float32 sau khi tạo SFTTrainer, trước `trainer.train()` (khi train fp16 hoặc QLoRA), in dtype ra log. Test `tests/test_sua_loi_fp16_t4.py` (4 test, có test TRL thật cho thấy tham số LoRA thành bf16 rồi được đổi về float32 và train được). Gộp vào PR #10. |
| 25/9 | M15 (dở) | Chủ repo chạy lại `smoke` sau PR #10: train xong 125 bước. Số đo thật (`so_do/smoke.json`): train 14,9 phút (ước tính 13,3), 5,36 TFLOPS, VRAM 2,5 GB (ước tính 1,3), chấm 2,2 và 2,6 phút, điểm 16 → 14/30. Sửa `train_tflops` T4 6 → 5,4, tính lại thời gian trong README và `docs/TRAIN_COLAB.md`, thêm bảng số đo thật. Lỗi gặp: `duration_s` của eval tính cả thời gian tải và nạp model (model nạp lười ở câu đầu), nên `token_overhead_s` đề xuất (0,054) không dùng được; sửa: eval nạp model trước khi bấm giờ, ghi `load_s`, `calibrate` đánh dấu báo cáo cũ. 240 test qua, compileall và secret-scan sạch. Tiếp theo: chờ `light` và agent. |
| 25/9 | M15 (dở, lượt 2) | Chủ repo gọi M15 và nhờ gộp PR #11 (đã gộp `e6fb23c`). `light` đã train xong 125 bước trên T4 (Colab ngắt sau bước 30, chạy tiếp được): 95 bước trong 66,7 phút, VRAM 8,2 GB (ước tính 3,8), chấm trước 17/30 trong 17,4 phút. Chưa có chấm sau và `so_do/light.json`. Sửa `train_tflops` 5,4 → 5,2; tính lại thời gian (light 121 phút); điền cột `light` trong README. Không sửa `TRAINING_FACTOR` vì sẽ làm hỏng test M2 (27B vừa 24 GB): chờ chủ repo. Phát hiện: Qwen3 dùng hết 512 token cho `<think>` nên code 1/8 (ghi Việc dở). 244 test qua. |
| 26/9 | M15 (dở, lượt 3) | Chủ repo: gộp PR #12 (đã gộp), đồng ý sửa công thức VRAM. Không nhân hệ số 3,61 cho mọi model (27B thành 57 GB, quá cao); thêm `KBIT_OVERHEAD_GB` × √(tỷ tham số) cho model nén (embedding float32, logits), đo trên `light`: light 9,2 GB, smoke 3,2 GB, 27B 34,6 GB. `calibrate` đề xuất hằng số mới cho model nén. Sửa test M2 (27B QLoRA trên 24 GB, không quá 48) và các test M15 liên quan; cập nhật README, cấu hình, ARCHITECTURE. 244 test qua. Tiếp theo: chờ chấm sau của `light` và agent. |
| 26/9 | M15 (dở, lượt 4) | Chủ repo chạy xong `smoke` và `light` (Bước 10, 12). `light`: 17 → 19/30, tool_use 7/8 → 3/8, chấm sau chỉ 2,5 phút vì bỏ `<think>`. Hệ số VRAM 2,07 của `light` là từ lần chạy 0 bước (chỉ nạp model), không dùng; sửa lệnh train để không ghi đè số đo khi 0 bước, `calibrate` bỏ qua VRAM 0 bước. `token_overhead_s` giữ nguyên (ước tính 17,2 khớp đo 17,4). Gỡ torchao ở Bước 3. Viết đề xuất giữ tool_use (README, M19/M20). Gộp vào PR #13. 247 test qua. Còn chờ: agent. |
| 26/9 | M15 (xong) | Chủ repo chạy `agent_colab` trên T4: 4/5 (80%) trong 11,1 phút; không đạt `tong-cot-csv` (awk có `>` bị TerminalTool từ chối, model không thử lại bằng `cat`, trả lời 30 thay vì 87). Điền cột Agent trong README, lưu kết quả vào fixture, thêm `RealAgentMeasurementTests`. Tiêu chí 3 đạt, M15 xong 4/4. 249 test qua, compileall và secret-scan sạch. Tiếp theo: mốc chủ repo chọn (M17–M21). |
