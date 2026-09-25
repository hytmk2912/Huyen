# Bộ nhớ làm việc

TUẦN 2

**XONG TUẦN 2 – chờ merge.** Chủ repo yêu cầu gộp PR #8 vào `main` ngay khi M14 xong (25/9).

Skill `lam-moc` đọc file này ở đầu mỗi lượt và cập nhật ở cuối lượt. Viết ngắn (dưới 80 dòng). Kế hoạch và tiêu chí: `TASKS.md` (tuần 2 ở đầu file). Nhật ký tuần 1: `archive/memory-tuan-1.md`.

## Tồn tuần 1
- Không có mốc tồn: M1–M7 đều xong, đã gộp vào `main` (PR #6 `63bdc61`, PR #7 `09c93f3`).

## Checklist tuần 2
- [x] M8 Gộp repo Agent, phần 1: đưa code runtime vào (xong 24/9)
- [x] M9 Gộp repo Agent, phần 2: tool chạy lệnh an toàn (xong 24/9)
- [x] M10 Notebook train trên Colab free (0.5B) (xong 25/9)
- [x] M11 Colab cho Qwen3-4B + hướng dẫn iPhone (xong 25/9)
- [x] M12 Chất lượng dữ liệu (xong 25/9)
- [x] M13 Agent chạy model thật trên Colab (xong 25/9)
- [x] M14 Tổng kết tuần 2 (xong 25/9)

## Mốc đang làm
- Không có mốc dở. Tuần 3 chưa có kế hoạch: chủ repo chọn trong các đề xuất dưới đây, rồi mới ghi vào `TASKS.md`.

## Tiến độ từng phần (sau tuần 2)
Cách tính: % = số hạng mục đã xong / tổng số hạng mục của phần đó. "Xong" nghĩa là có test; với việc cần chạy thật (GPU, Colab, model thật) thì phải đã chạy thật.

| Phần | % | Đã xong | Còn thiếu |
| --- | ---: | --- | --- |
| Mốc tuần 2 | 100% | 7/7 mốc, 206 test | — |
| Dữ liệu | 100% | 4/4: preset và trộn, chặn trùng eval, lọc chất lượng, chạy thật 2000 dòng | — |
| Fine-tune | 40% | 2/5: code full/LoRA/QLoRA, chạy thật LoRA trên CPU | smoke và light trên Colab T4, primary trên GPU 24 GB |
| Eval | 67% | 2/3: 30 câu 4 cách chấm và so sánh trước/sau, chạy thật với model tí hon | chấm model smoke/light thật |
| Model qua server local | 50% | 1/2: adapter OpenAI (test bằng server giả) | chạy với Ollama hoặc llama.cpp thật |
| Agent và runtime | 80% | 4/5: vòng lặp, TerminalTool an toàn, gateway, 5 nhiệm vụ mẫu | chạy với model thật (`agent_colab`) |
| Notebook Colab | 50% | 3/6: viết và test dry-run `train_colab` (smoke, light) và `agent_colab` | chạy thật cả 3 trên Colab |
| Tài liệu | 100% | 4/4: README, ARCHITECTURE, TRAIN_COLAB, GOP_AGENT | ghi số đo thật sau khi chạy |

## Việc chủ repo tự làm
1. Tạo token Hugging Face quyền **Write**, thêm vào Colab Secrets tên `HF_TOKEN` và bật Notebook access.
2. Chạy `notebooks/train_colab.ipynb` với `smoke` (khoảng 30 phút; hướng dẫn `docs/TRAIN_COLAB.md`), gửi lại thời gian thật và bảng so sánh ở Bước 10.
3. Chạy lại với `light` (khoảng 2 giờ; Colab ngắt thì Run all lại), gửi thời gian, VRAM và bảng so sánh.
4. Chạy `notebooks/agent_colab.ipynb`, gửi tỉ lệ thành công và trace của nhiệm vụ không đạt.
5. Archive repo Agent cũ `huyenytmk2912/agent`. Nếu `hytmk2912/Agent` có code mới hơn `78a3e25`, cấp quyền để gộp thêm.
6. Quyết định PR #4 (đổi 3 model phụ sang Huihui Qwen3 4B/8B/14B).
7. Việc còn từ tuần 1: sửa mô tả repo trên GitHub ("Train Từ Số 0"); đổi mật khẩu máy chủ cũ còn trong lịch sử commit; cân nhắc chuyển repo sang Private; đọc lại giấy phép 3 dataset preset.

## Đề xuất 7 mốc tuần 3 (chưa làm; giống cuối README)
1. M15 Số đo thật: sửa hằng số trong `estimate.py` và `vram.py` theo kết quả các notebook chủ repo chạy; ghi bảng số đo vào README.
2. M16 Notebook bền hơn: dừng Run all khi lệnh `!python` lỗi (thử `_exit_code` trên Colab); lưu báo cáo chấm trước lên repo HF để chạy lại không phải chấm lại.
3. M17 Model chính (ảnh + chữ): chỉ gắn LoRA vào phần ngôn ngữ; đổi `torch_dtype` sang `dtype`.
4. M18 Gộp adapter vào model gốc, xuất GGUF, cho agent chạy bằng model vừa train và so với model gốc.
5. M19 Mở rộng eval: 10 nhiệm vụ agent nhiều bước, câu dùng công cụ; tự chấm sau mỗi lần train.
6. M20 Dữ liệu: chặn gần trùng train/eval bằng MinHash của M12; thêm một preset tiếng Việt (đọc giấy phép trước).
7. M21 Tổng kết tuần 3.

## Việc dở
- Nhánh làm việc: `claude/nhiem-vu-tuan-2` (tạo từ `main` tại `09c93f3`); PR #8.
- Hằng số ước tính (`estimate.py`: 6 TFLOPS, 190 GB/s) và các con số thời gian trong tài liệu chưa đo trên T4 thật (xem M15).

## Lỗi còn tồn
| Lỗi | Nơi |
| --- | --- |
| Chưa chạy thật trên GPU hay model thật: QLoRA, model ảnh + chữ, 2 notebook Colab, agent với Ollama + qwen3:4b. `target_modules: "all-linear"` có thể gắn LoRA vào phần xử lý ảnh. | `adapters.py`, `finetune.py`, `notebooks/` |
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
| 25/9 | M12 | Xong 3/3. `local_ai/data/quality.py` (thư viện chuẩn) với 4 bộ lọc: ngôn ngữ ưu tiên tiếng Việt, độ dài, lặp (xét từng lượt, bỏ code và LaTeX), gần trùng (MinHash tự viết một hoán vị + LSH, so lại bằng Jaccard thật). Ngưỡng ở `configs/datasets/quality.json`; `hf-sft` bật mặc định (`--no-quality` để tắt), `build --quality`. Mục `quality` trong `manifest.json` ghi trước/sau lọc. Fixture cho từng bộ lọc, có dòng đối chứng phải giữ. Lỗi gặp: MinHash 128 hoán vị mất 21 giây cho 2000 dòng, đổi sang kiểu một hoán vị (1,5 giây); bắt nhầm `\frac`/`\cdot`, "1 cm 2 cm", hội thoại nhắc lại công thức qua nhiều lượt, tiếng Việt trong preset code; đã sửa. Trên 2000 dòng thật chỉ loại 1 dòng. 189 test qua, compileall và secret-scan sạch. Tiếp theo: M13. |
| 25/9 | M13 | Xong 3/3. `notebooks/agent_colab.ipynb`: Ollama ghim 0.34.4 (lấy từ link tải bản mới nhất; bản cài `.tar.zst` cần zstd), `ollama serve` ở nền, `qwen3:4b` (đã kiểm tra tag còn trên kho Ollama), mục `ollama-colab`. Lệnh mới `local_ai.agents.tasks`: 5 nhiệm vụ (calculator, `ls`, `cat`, `cat` rồi calculator), mỗi nhiệm vụ có thư mục làm việc riêng, đạt khi đúng đáp án và đã gọi công cụ cần dùng; in trace và tỉ lệ. Phát hiện prompt của agent chưa nói có công cụ nào, nên model thật không gọi được: thêm mô tả công cụ vào `ToolRegistry`, gửi kèm lịch sử quan sát và mẫu JSON. Test chạy 5 nhiệm vụ qua server OpenAI giả: 5/5. 200 test qua, compileall và secret-scan sạch. Chưa chạy với Ollama thật. Tiếp theo: M14. |
| 25/9 | M14 | Xong 4/4. README: bảng trạng thái 2 tuần, 2 nút Colab, lộ trình đề xuất M15–M21; ARCHITECTURE: luồng Colab và agent, kiểm thử theo mốc, phần chưa kiểm chứng; `memory.md`: % từng phần, việc chủ repo tự làm, 7 đề xuất tuần 3. `tests/test_m14_summary.py` giữ README khớp code (tìm ra README thiếu `local_ai.training.hub`, đã thêm). Lỗi gặp: lúc chèn mục agent ở M13 suýt làm mất tiêu đề lộ trình, test M7 bắt được. 206 test qua, compileall và secret-scan sạch. XONG TUẦN 2. |
