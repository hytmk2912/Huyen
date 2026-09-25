# Bộ nhớ làm việc

TUẦN 2

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng). Kế hoạch và tiêu chí: `TASKS.md` (tuần 2 ở đầu file). Nhật ký tuần 1: `archive/memory-tuan-1.md`.

## Tồn tuần 1
- Không có mốc tồn: M1–M7 đều xong, đã gộp vào `main` (PR #6 `63bdc61`, PR #7 `09c93f3`).
- Việc cần GPU hoặc server thật (không phải mốc): chạy smoke → light → primary trên GPU; thử adapter server local với Ollama/llama.cpp thật.

## Checklist tuần 2
- [x] M8 Gộp repo Agent, phần 1: đưa code runtime vào (xong 24/9)
- [x] M9 Gộp repo Agent, phần 2: tool chạy lệnh an toàn (xong 24/9)
- [x] M10 Notebook train trên Colab free (0.5B) (xong 25/9)
- [x] M11 Colab cho Qwen3-4B + hướng dẫn iPhone (xong 25/9)
- [x] M12 Chất lượng dữ liệu (xong 25/9)
- [ ] M13 Agent chạy model thật trên Colab
- [ ] M14 Tổng kết tuần 2

## Mốc đang làm
- Không có mốc dở. Mốc kế tiếp: **M13** (`notebooks/agent_colab.ipynb`: cài Ollama, kéo model nhỏ, agent gọi qua adapter OpenAI; 5 nhiệm vụ mẫu bằng calculator và `TerminalTool`, in trace và tỉ lệ thành công; test dry-run).

## Việc dở
- Nhánh làm việc: `claude/nhiem-vu-tuan-2` (tạo từ `main` tại `09c93f3`).
- PR #4 (`claude/expand-model-training-repo-v2yzyr`) vẫn mở, chờ chủ repo quyết định.
- Repo Agent: `hytmk2912/Agent` không truy cập được từ phiên (riêng tư hoặc chưa cấp quyền); đã lấy bản công khai `huyenytmk2912/agent` tại `78a3e25`. Nếu tên mới có code mới hơn thì cần chủ repo cấp quyền để gộp thêm. Gộp xong ở M9: chủ repo nên archive repo Agent cũ.
- Notebook Colab (ngoài `TASKS.md`, chờ chủ repo quyết định):
  - lệnh `!python` lỗi không dừng Run all, nên các ô sau lỗi theo; có thể kiểm tra `_exit_code` sau mỗi lệnh, nhưng cần thử trên Colab thật;
  - chạy lại sau khi Colab ngắt thì chấm lại model gốc (`light` mất thêm khoảng 17 phút); có thể lưu báo cáo chấm trước lên repo HF.
- Hằng số ước tính thời gian (`local_ai/training/estimate.py`: 6 TFLOPS, băng thông 190 GB/s) chưa đo trên T4 thật; chạy thật xong thì sửa lại.
- Việc chủ repo tự làm: sửa mô tả repo trên GitHub ("Train Từ Số 0"); đổi mật khẩu máy chủ cũ còn trong lịch sử commit; cân nhắc chuyển repo sang Private; đọc lại giấy phép 3 dataset preset.

## Lỗi còn tồn (từ tuần 1)
| Lỗi | Nơi |
| --- | --- |
| Chưa chạy thật trên GPU: QLoRA, model ảnh + chữ, notebook Colab (M10, M11); `target_modules: "all-linear"` có thể gắn LoRA vào phần xử lý ảnh. | `local_ai/models/adapters.py`, `local_ai/training/finetune.py` |
| transformers 5.17 cảnh báo `torch_dtype` đã cũ, nên đổi sang `dtype`. | `adapters.py`, `finetune.py` |

## Nhật ký tuần 2
| Ngày | Lượt | Kết quả |
| --- | --- | --- |
| 24/9 | Thiết lập | Tạo nhánh `claude/nhiem-vu-tuan-2` từ `main` (`09c93f3`); chuyển nhật ký tuần 1 sang `archive/memory-tuan-1.md`; thêm kế hoạch M8–M14 vào đầu `TASKS.md`; cập nhật skill `lam-moc` và `CLAUDE.md` cho tuần 2 (quy tắc notebook, Colab Secrets). |
| 24/9 | M8 | Xong 4/4. Clone `huyenytmk2912/agent` (`78a3e25`, 4 file, không có test). Viết lại runtime bằng thư viện chuẩn trong `local_ai/runtime/` (`run_command` chạy lệnh dạng list không qua shell, dừng cả nhóm tiến trình khi quá giờ; `JobQueue` có giới hạn; `require_token` so sánh an toàn). Cất code gốc ở `archive/agent-goc/`. `docs/GOP_AGENT.md` ghi 7 rủi ro bảo mật của bản gốc. Lỗi gặp: test tìm chữ `shell=True` bắt nhầm docstring, đổi sang kiểm tra bằng AST; lần chạy đầu 57 giây do phiên mới (lần sau 10 giây). 133 test qua, compileall và secret-scan sạch. Tiếp theo: M9. |
| 24/9 | M9 | Xong 5/5. `TerminalTool` (allowlist ở `configs/tools/terminal.json` đúng như Agent gốc, tắt mặc định, không shell, timeout, log JSONL, chặn ký tự shell, tham số nguy hiểm, đường dẫn ra ngoài thư mục làm việc) và `register_terminal` cho agent. Gateway HTTP bằng `http.server` (tắt mặc định, chỉ 127.0.0.1, token ≥ 16 ký tự từ biến môi trường); không đưa `/v1/execute` qua mạng. Lỗi gặp: khóa JSON dạng tên lệnh không qua được test "mọi khóa cấu hình phải được code đọc", nên đổi allowlist sang danh sách `{name: ...}`; tự rà thấy `grep -f/etc/passwd` lọt kiểm tra đường dẫn, đã chặn; 2 ca test viết sai, đã sửa. 144 test qua, compileall và secret-scan sạch. Tiếp theo: M10. |
| 24/9 | M9 (rà lại) | Chủ repo gõ lại "M9": M9 đã xong trên remote (`1214702`), rà thêm các cách lách `TerminalTool`. Tìm thấy `date -s2020-01-01`/`--set` lọt (đổi đồng hồ hệ thống nếu chạy root, ví dụ trên Colab) và tham số ngắn viết gộp (`date -us...`, `tail -qf`) lọt luật cũ; đã chặn, thêm 6 ca test; nâng `max_args` của `date` lên 3 để `date -u +%Y` chạy được. Tiếp theo: M10. |
| 25/9 | M10 | Xong 5/5. `notebooks/train_colab.ipynb` (sinh từ `notebooks/build.py`, 9 ô code, hợp lệ theo nbformat). Finetune thêm `--push-to-hub --hub-model-id` (`hub_strategy` checkpoint; chạy lại thì tải `last-checkpoint` về `_hub/` rồi train tiếp) và `dtype`; `local_ai/training/hub.py` (đẩy adapter lên repo riêng tư); `configs/training/colab_smoke.json`; mục `smoke-colab`; lệnh `local_ai.evaluation.compare`; `--dry-run` cho `hf-sft` và eval. README có nút Open in Colab. Lỗi gặp: chuỗi có dấu `\"` bị thoát sai khi ghi file (lỗi cú pháp), đã sửa; `help=` thiếu chữ có dấu, test thống nhất bắt được. Tự rà thấy checkpoint tải về nằm trong `output_dir` sẽ bị Trainer đẩy ngược lên Hub, nên chuyển sang `_hub/`. 160 test qua, compileall và secret-scan sạch. Chưa chạy trên Colab thật. Tiếp theo: M11. |
| 25/9 | M11 | Xong 4/4. `configs/training/colab_light.json` (Qwen3-4B, QLoRA 4bit fp16, batch 1, `max_length` 2048, VRAM ước tính 3,8 GB) và mục `light-colab`. Notebook có ô chọn model (form Colab smoke/light) và ô ước tính thời gian (lệnh mới `local_ai.training.estimate`, báo cả số bước đã train khi Colab từng ngắt). `docs/TRAIN_COLAB.md` hướng dẫn trên iPhone. Đo độ dài thật: stream 2000 dòng từ 3 preset (16 giây) và đếm token bằng tokenizer Qwen3 (không tải trọng số), ra trung bình 491 token/dòng khi cắt ở 1024; batch 4 đệm lên 820 token/dòng. Lỗi gặp: `apply_chat_template(tokenize=True)` của transformers 5 trả dict nên lần đếm đầu sai, đã đổi cách đếm. Test M10 sửa theo cấu trúc notebook mới, không bỏ kiểm tra nào. 171 test qua, compileall và secret-scan sạch. Chưa chạy trên Colab thật. Tiếp theo: M12. |
| 25/9 | M12 | Xong 3/3. `local_ai/data/quality.py` (thư viện chuẩn) với 4 bộ lọc: ngôn ngữ ưu tiên tiếng Việt, độ dài, lặp (xét từng lượt, bỏ code và LaTeX), gần trùng (MinHash tự viết một hoán vị + LSH, so lại bằng Jaccard thật). Ngưỡng ở `configs/datasets/quality.json`; `hf-sft` bật mặc định (`--no-quality` để tắt), `build --quality`. Mục `quality` trong `manifest.json` ghi trước/sau lọc. Fixture cho từng bộ lọc, có dòng đối chứng phải giữ. Lỗi gặp: MinHash 128 hoán vị mất 21 giây cho 2000 dòng, đổi sang kiểu một hoán vị (1,5 giây); bắt nhầm `\\frac`/`\\cdot`, "1 cm 2 cm", hội thoại nhắc lại công thức qua nhiều lượt, tiếng Việt trong preset code; đã sửa. Trên 2000 dòng thật chỉ loại 1 dòng. 189 test qua, compileall và secret-scan sạch. Tiếp theo: M13. |
