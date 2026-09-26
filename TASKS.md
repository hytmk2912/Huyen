# Nhiệm vụ tuần 3 (chủ repo chọn từng mốc)

Bắt đầu: 2026-09-25, từ `main` tại commit `12e86fc` (206 test chạy qua; compileall và secret-scan sạch). Nhánh làm việc: `claude/nhiem-vu-tuan-jixv6i`.

Cách làm giống tuần 2: mỗi lượt **tối đa 1 mốc** bằng skill `lam-moc`; mốc chỉ **Xong** khi đạt mọi tiêu chí và `python -m local_ai.check` (không `--allow-skip`) xanh; khi đó tự gộp PR. Khác tuần 2: 7 mốc M15–M21 (cuối README) chỉ là đề xuất. Chủ repo chọn mốc nào thì mốc đó mới được thêm tiêu chí vào đây.

## Bảng tiến độ tuần 3

| Mốc | Nội dung | Trạng thái | Tiến độ | Bằng chứng |
| --- | --- | --- | --- | --- |
| M15 | Số đo thật trên Colab | **Xong** | 4/4 (100%) | `tests/test_m15_measurements.py` (21 test) |
| M16 | Notebook bền hơn | **Xong** | 4/4 (100%) | `tests/test_m16_notebook_resilience.py` (13 test, 2 test thêm ở lượt rà soát tuần 3) |
| M17 | Model chính (ảnh + chữ): LoRA chỉ gắn vào phần ngôn ngữ | **Xong** | 4/4 (100%) | `tests/test_m17_multimodal_lora.py` (5 test) |
| M19 | Chấm công bằng hơn | **Xong** | 8/8 (100%) | `tests/test_m19_cham_cong_bang.py` (23 test) |
| M20 | Dữ liệu giữ khả năng gọi công cụ | **Chưa làm** (chọn 26/9) | 0/4 (0%) | — |
| **Tổng** | 5 mốc đã chọn (M18, M21 chưa chọn) | 4 **Xong**, 1 chưa làm | 4/5 mốc (80%) | `python -m local_ai.check`: 294 test chạy qua, không test nào bị bỏ qua; compileall và secret-scan sạch (26/9) |

Rà soát M1–M16 (25/9, chủ repo yêu cầu, không phải mốc): sửa `torch_dtype` → `dtype`, secret-scan bắt thêm kiểu mật khẩu shell từng lộ, `.gitignore` thêm `*.bin` và `.ruff_cache/`, ghi giấy phép dataset vào `docs/GIAY_PHEP_DATASET.md`. Bằng chứng: `tests/test_ra_soat_m1_m16.py` (6 test); 232 test chạy qua. Việc chỉ chủ repo làm được: mục "Việc chủ repo tự làm" trong `memory.md`.

Rà soát tuần 3 (26/9, chủ repo yêu cầu, không phải mốc): tài liệu khớp thực tế sau khi chạy thật trên Colab T4; tuần 1–2 của file này chép nguyên văn sang `archive/tasks-tuan-1-2.md`; `memory.md` tối đa 5.000 ký tự (nhật ký chi tiết ở `archive/memory-tuan-3.md`); lệnh mới `python -m local_ai.check` (test bị bỏ qua vì thiếu thư viện thì không xanh); hàm `run` của `local_ai/colab.py` đóng pipe và in nốt chữ còn trong decoder; ghi tiêu chí M19, M20. Bằng chứng: `tests/test_ra_soat_tuan_3.py`, `RunCleanupTests` trong `tests/test_m16_notebook_resilience.py`.

Sửa 2 lỗi bảo mật (26/9, chủ repo yêu cầu và đã thử thật, không phải mốc): (1) TerminalTool đọc được repo cha qua git, vì thư mục làm việc của agent nằm trong repo và git tự tìm `.git` ở thư mục cha; nay lệnh chạy với `GIT_CEILING_DIRECTORIES` là thư mục cha của thư mục làm việc. (2) PythonSandbox để lộ `HF_TOKEN` (notebook đặt vào `os.environ`); nay sandbox và lệnh của TerminalTool chỉ nhận môi trường tối thiểu (PATH, ngôn ngữ/mã hóa, HOME tạm). Bằng chứng: `tests/test_bao_mat_sandbox_terminal.py` (5 test). Ghi thêm tiêu chí M19 (5–7) và M20 (3).

Sửa lỗi Bước 8 trên T4 (25/9, chủ repo gặp khi chạy `smoke`): TRL đổi tham số LoRA sang bf16 nên train fp16 lỗi ở bước đầu; nay tham số được train được đổi về float32 trước khi train và dtype được in ra log. Bằng chứng: `tests/test_sua_loi_fp16_t4.py` (4 test).

## M15: Số đo thật trên Colab
Mục tiêu: sửa các hằng số ước tính (`local_ai/training/estimate.py`, `local_ai/models/vram.py`) theo số đo thật khi chủ repo chạy notebook, và ghi bảng số đo thật vào README. Hiện notebook không đo VRAM, và chủ repo khó chép số từ điện thoại, nên cần làm phần ghi số đo trước.
- [x] 1. Ghi số đo khi chạy thật:
  - lệnh train ghi `measurements.json` (GPU, VRAM đỉnh, thời gian, bước bắt đầu/kết thúc, số token của lần chạy), ghi trước khi lưu adapter nên được đẩy lên Hub cùng adapter;
  - báo cáo eval ghi `duration_s`, `output_chars`;
  - báo cáo agent ghi `duration_s`, và in thời gian cạnh tỉ lệ thành công.

  Bằng chứng: `RealCpuTrainingTests` train thật 2 bước rồi chạy tiếp 1 bước trên CPU (bước bắt đầu 2, token chỉ tính lần chạy tiếp); `FakeGpuTrainingTests` (VRAM, tên GPU, file có trước khi lưu adapter); `ReportDurationTests`.
- [x] 2. Lệnh `python -m local_ai.training.calibrate`:
  - so số đo thật với ước tính, in bảng và hằng số đề xuất (`train_tflops`, `TRAINING_FACTOR`, `token_overhead_s`);
  - lưu `so_do.json`, tùy chọn đẩy lên `so_do/<model>.json` của repo Hugging Face riêng tư;
  - notebook train có ô **Bước 12** chạy lệnh này.
  - Hệ số VRAM đo từ model nhỏ bị đánh dấu là không dùng được: ở model 0.5B phần logits lớn hơn trọng số nhiều lần, nên hệ số tính ra rất lớn, ví dụ 18,87 với số đo mẫu.
  - GPU chưa có hồ sơ (ví dụ CPU, L4) thì chỉ ghi số đo, không đề xuất.

  Bằng chứng: `CalibrateTests`, dùng SỐ ĐO MẪU trong `tests/fixtures/measurements/`, không phải số đo thật.
- [x] 3. Có số đo thật từ chủ repo (smoke, light, agent): sửa hằng số trong `estimate.py` và `vram.py`; README có bảng số đo thật. **Đạt (26/9):**
  - [x] `smoke` (Colab T4, 25/9, sau khi sửa lỗi Bước 8): train 125 bước trong 14,9 phút (ước tính 13,3), VRAM 2,5 GB (ước tính 1,3), chấm 2,2 và 2,6 phút, điểm 16/30 → 14/30. File `so_do/smoke.json` chép vào `tests/fixtures/measurements/that_smoke_t4_2026-09-25.json`.
  - Đã sửa: `GPUS["T4"].train_tflops` 6 → 5,4 (đo 5,36); thời gian trong README và `docs/TRAIN_COLAB.md` tính lại (`smoke` 23 phút, `light` 118 phút). README có bảng số đo thật (cột `light` và agent ghi "chưa chạy").
  - Chưa sửa: `TRAINING_FACTOR` (số đo model nhỏ không dùng được, chờ `light`); `token_overhead_s` (thời gian chấm lần này tính cả thời gian tải và nạp model). Đã sửa cách đo: lệnh eval nạp model trước khi bấm giờ và ghi `load_s` riêng; `calibrate` đánh dấu số đo từ báo cáo cũ là không dùng được.
  - [x] `light` (Colab T4, 25–26/9): đủ số đo. Chấm sau khi train (26/9): 19/30 trong 2,5 phút, nhưng tool_use 7/8 → 3/8 và model bỏ phần suy nghĩ `<think>` (đề xuất cách giữ tool_use ở README). Lần chạy lại 26/9 không train bước nào nên `measurements.json` trên Hub bị ghi đè bằng số đo 0 bước (VRAM 4,68 GB chỉ là lúc nạp model, hệ số 2,07 trong `so_do/light.json` không dùng được); đã sửa: lệnh train giữ số đo của lần đã train, `calibrate` bỏ qua VRAM của lần chạy 0 bước. Lần train 25/9: Colab ngắt sau bước 30, lần chạy tiếp train nốt 95 bước trong 66,7 phút (42 giây/bước); VRAM 8,2 GB (ước tính 3,8); chấm trước 17/30 trong 17,4 phút (ước tính tối đa 17,2). Số đo chép vào `tests/fixtures/measurements/that_light_t4_2026-09-25.json`.
    - Đã sửa: `train_tflops` 5,4 → 5,2 (smoke 5,36, light 5,17): ước tính của cả 2 model lệch dưới 5%; thời gian trong tài liệu tính lại (`light` 121 phút). README điền cột `light`.
    - VRAM (chủ repo đồng ý ngày 26/9, "tùy chỉnh sao cho hợp"): không nhân hệ số 3,61 cho mọi model (27B sẽ thành 57 GB, quá cao), mà cộng thêm phần cho model nén `KBIT_OVERHEAD_GB` × √(tỷ tham số) = 2,67 × √(tỷ tham số) GB (embedding float32, logits), đo trên `light`. Ước tính: `light` 9,2 GB (đo 8,2 + CUDA context), `smoke` 3,2 GB (đo 2,5), 27B 34,6 GB (trước 20,5; chưa đo, cần đo ở M17). `calibrate` đề xuất `KBIT_OVERHEAD_GB` cho model nén. Test M2 `test_numbers_for_27b` sửa theo (27B QLoRA trên 24 GB, không quá 48 GB).
    - `token_overhead_s` giữ nguyên, có số đo làm căn cứ: ước tính chấm tối đa của `light` 17,2 phút khớp đo thật 17,4 phút; lần chấm sau (câu trả lời ngắn) cho 0,12 giây/token vì chi phí cố định mỗi câu, không đại diện.
  - [x] agent (Colab T4, 26/9, Ollama + `qwen3:4b`): 4/5 nhiệm vụ đạt (80%) trong 11,1 phút. Không đạt `tong-cot-csv`: model chạy `awk ... 'NR > 1 ...'`, TerminalTool từ chối đúng quy tắc (ký tự `>`), model không thử lại bằng `cat` mà trả lời 30 thay vì 87. Kết quả chép từ ảnh chụp vào `tests/fixtures/measurements/that_agent_t4_2026-09-26.json`; README điền cột Agent.

  Bằng chứng: `RealSmokeMeasurementTests` (4 test: hằng số T4 nằm giữa 2 số đo thật, ước tính `smoke` lệch dưới 5%, eval nạp model trước khi bấm giờ, README có bảng số đo); `RealLightMeasurementTests` (4 test: lần chạy tiếp chỉ đếm bước và token của nó, ước tính train và chấm lệch dưới 5%, công thức VRAM QLoRA theo `light`, README có cột `light` và đề xuất giữ tool_use); `ZeroStepRerunTests` (2 test: lần chạy 0 bước không ghi đè số đo, `calibrate` bỏ qua VRAM của nó); `RealAgentMeasurementTests` (2 test: kết quả khớp bộ nhiệm vụ, bảng `calibrate` và README có dòng agent, bảng số đo không còn ô "chưa chạy"). Gỡ torchao ở Bước 3: `tests/test_sua_loi_colab.py`.
- [x] 4. Test xanh, kể cả chạy lệnh mới của notebook bằng `--dry-run`. Test M10/M11 được sửa theo notebook mới (thêm ô Bước 12; không bỏ kiểm tra nào); test M14 đọc đúng phần tuần 2 khi tuần 3 được thêm lên đầu `TASKS.md`. 215 test chạy qua; compileall và secret-scan sạch. Khi xong mốc (26/9): 249 test chạy qua; compileall và secret-scan sạch.

## M17: Model chính (ảnh + chữ): LoRA chỉ gắn vào phần ngôn ngữ
Chủ repo chọn ngày 26/9. Mục tiêu: model chính (`huihui-ai/Huihui-Qwen3.8-27B-abliterated`, kiến trúc Qwen3.5 ảnh + chữ) được train bằng dữ liệu chỉ có chữ, nên LoRA chỉ gắn vào phần ngôn ngữ, không gắn vào phần xử lý ảnh (vision encoder, merger/projector). Hiện `target_modules: "all-linear"` gắn LoRA vào cả phần xử lý ảnh. Việc đổi `torch_dtype` sang `dtype` đã làm ở lượt rà soát M1–M16. Train thật model 27B cần GPU 40–48 GB nên không thuộc mốc này.
- [x] 1. Model `kind: "multimodal"`: LoRA loại trừ phần xử lý ảnh (`exclude_modules` của PEFT); model chữ giữ nguyên. `--dry-run` của lệnh train in ra phạm vi LoRA (gắn vào đâu, loại trừ gì).
  Bằng chứng: `LoraScopeTests` (3 test: chỉ model multimodal loại trừ, biểu thức `VISION_MODULES` khớp phần xử lý ảnh của Qwen3.5, LLaVA, Gemma 3 và không khớp phần ngôn ngữ, `--dry-run` in mục `lora`).
- [x] 2. Chứng minh trên model tí hon cùng kiến trúc với model chính (Qwen3.5 ảnh + chữ, khởi tạo ngẫu nhiên, không tải gì): không có LoRA nào trong phần xử lý ảnh; mọi lớp Linear của phần ngôn ngữ (attention thường, linear attention, MLP) đều có LoRA; `lm_head` không có.
  Bằng chứng: `TinyQwen35Tests.test_all_linear_touches_vision_and_exclusion_fixes_it`. Test cũng cho thấy lỗi cũ: chỉ dùng `all-linear` thì LoRA gắn vào 10 lớp Linear của phần xử lý ảnh.
- [x] 3. Train thật 2 bước trên CPU bằng lệnh train của repo với model tí hon đó và dữ liệu chữ: xong, lưu adapter; `adapter_config.json` ghi phần loại trừ; nạp lại adapter vào model gốc được và không có LoRA trong phần xử lý ảnh.
  Bằng chứng: `TinyQwen35Tests.test_real_cpu_training_keeps_vision_without_lora` (nạp bằng `AutoModelForMultimodalLM`, đúng lớp `Qwen3_5ForConditionalGeneration` của model chính; chạy khoảng 2 giây).
- [x] 4. README (mục train trên GPU, lộ trình), `docs/ARCHITECTURE.md` cập nhật. 254 test chạy qua; compileall và secret-scan sạch.

## M19: Chấm công bằng hơn
Chủ repo chọn ngày 26/9; làm trước M20 (chủ repo gọi `/lam-moc`). Mục tiêu: sửa các điểm chấm chưa công bằng thấy khi chạy thật trên Colab (M15): Qwen3 "suy nghĩ" hết 512 token trước khi viết code; báo cáo chấm sau khi train không được lưu; chỉ 8 câu tool_use; agent không thử lại khi TerminalTool từ chối lệnh.
- [x] 1. Tùy chọn tắt chế độ suy nghĩ của Qwen3 khi chấm (`enable_thinking=False` truyền vào `apply_chat_template`). Cài đặt này nằm trong `eval_settings`, để không dùng lại nhầm báo cáo cũ trên Hub. Notebook dùng cùng cài đặt cho chấm trước và chấm sau.

  Đã làm: lệnh eval có `--no-thinking`; `ModelConfig.enable_thinking` truyền vào chat template (model chữ và model ảnh + chữ) và gửi `chat_template_kwargs` cho server; Bước 7 và Bước 9 đều dùng `--no-thinking`. Bằng chứng: `ThinkingTests` (5 test, gồm báo cáo cũ trên Hub không được dùng lại khi bật `--no-thinking`), `ReproducibleEvalTests.test_no_thinking_reaches_the_chat_template` (chat template kiểu Qwen3 nhận khối suy nghĩ rỗng).
- [x] 2. Bước 9 (chấm sau) đẩy báo cáo lên Hub ở `eval/sau` và luôn chấm lại (thêm tùy chọn `--no-reuse` cho lệnh eval).

  Bằng chứng: `AfterTrainingReportTests` (2 test: `--no-reuse` không tải báo cáo cũ, chấm lại rồi đẩy lên). Test M16 `test_eval_before_training_is_cached_on_hub` trước kiểm tra Bước 9 không có `--hub-repo`; nay kiểm tra có `--hub-path eval/sau --no-reuse` (vẫn giữ ý: chấm sau luôn chấm lại).
- [x] 3. Câu tool_use trong bộ chấm tăng từ 8 lên ít nhất 16.

  Đã làm: thêm 8 câu (4 Việt, 4 Anh: calculator, read_file, search, không cần công cụ), bộ chấm 30 → 38 câu. `calibrate` ước tính thời gian chấm theo số câu của từng báo cáo, nên so sánh với số đo thật 30 câu (M15) vẫn đúng. Thời gian trong README và `docs/TRAIN_COLAB.md` tính lại theo 38 câu. Bằng chứng: `ToolUseCasesTests` (3 test, gồm mỗi câu tool_use từ chối đáp án của câu khác). Test cũ khóa số câu (M5 `28–35 câu`, `30/30`; M11 `eval_cases` 30; `test_consistency` `30/30`) sửa theo 38 câu, không bỏ kiểm tra nào.
- [x] 4. TerminalTool từ chối lệnh thì kèm gợi ý cách khác (các lệnh được phép; chạy từng lệnh, không pipe hay redirect). Thêm 1 nhiệm vụ agent phải thử lại sau khi bị từ chối; test bằng `--scripted` hoặc server giả.

  Đã làm: lỗi từ chối kèm "Gợi ý" tiếng Việt và "Hint" tiếng Anh cho model (trừ khi công cụ đang tắt); nhiệm vụ `thu-lai-khi-bi-tu-choi` (chạy `wc -l` bị từ chối, rồi thử lại bằng `cat`), chỉ đạt khi có lệnh bị từ chối rồi một lần gọi TerminalTool thành công. Bằng chứng: `TerminalHintTests` (3 test: `--scripted`, model không thử lại thì không đạt, server OpenAI giả nhận được gợi ý). Test M13 và M15 khóa 5 nhiệm vụ sửa theo 6 nhiệm vụ (số đo thật 26/9 vẫn so với 5 nhiệm vụ đầu).
- [x] 5. Chấm lặp lại được. Hiện `generate()` dùng cấu hình sinh chữ mặc định của Qwen (lấy mẫu ngẫu nhiên, không seed) nên chạy lại ra điểm khác. Model transformers mặc định greedy (`do_sample=False`); model qua server gửi `temperature` 0 và `seed` cố định. Cách sinh chữ và revision model nằm trong `eval_settings` và trong báo cáo. Test: chấm 2 lần bằng model tí hon ra cùng kết quả.

  Đã làm: `ModelConfig` có `temperature` (mặc định 0: greedy, ghi đè `do_sample` của model), `seed`, `generation_settings`; server nhận `temperature` 0 và `seed`; `report.json` có khóa `settings` (model, nguồn, revision, `max_new_tokens`, cách sinh chữ, mã băm bộ câu hỏi), `report.md` có dòng "Cài đặt". Bằng chứng: `ReproducibleEvalTests` (4 test, model tí hon với `generation_config` lấy mẫu như Qwen: mặc định cũ ra kết quả khác nhau, nay 2 lần chấm giống hệt), `GenerationConfigTests` (2 test). Test M3 (payload gửi server) và M16 (khóa của `settings`) sửa theo cài đặt mới.
- [x] 6. Chấm nhiệm vụ agent so số theo giá trị. Hiện `answer_matches` chỉ kiểm tra chuỗi con nên "187" được tính đúng khi cần "87". Sửa để 187 hay 87,5 không khớp 87; 337500.0 vẫn khớp 337500.

  Bằng chứng: `NumericAnswerTests` (2 test; mục chữ như `DH-4827` vẫn so chuỗi con).
- [x] 7. Ghim revision (mã commit) cho các model transformers trong `configs/models/platform.json`; mục `-lora`, `-colab` dùng cùng revision với model gốc. Không lấy được mã commit thì ghi vào "Việc dở", không đoán.

  Mã commit lấy ngày 26/9 từ API Hugging Face (`/api/models/<repo>/revision/main`, chỉ đọc thông tin, không tải trọng số): Qwen2.5-0.5B-Instruct `7ae5576…`, Qwen3-4B `1cfa9a7…`, Qwen2.5-Coder-7B-Instruct `c03e6d3…`, Huihui-Qwen3.8-27B-abliterated `739e3c5…`. Bằng chứng: `PinnedRevisionTests` (2 test: đủ 40 ký tự hex, biến thể dùng cùng revision, train và eval dùng revision này).
- [x] 8. `python -m local_ai.check` xanh: 294 test chạy qua, không test nào bị bỏ qua; compileall và secret-scan sạch.

## M20: Dữ liệu giữ khả năng gọi công cụ
Chủ repo chọn ngày 26/9; làm sau M19. Mục tiêu: sau khi train, `light` tụt tool_use 7/8 → 3/8 vì dữ liệu train không có dòng gọi công cụ; dữ liệu train mới giữ được khả năng này và không gần trùng với bộ chấm.
- [ ] 1. `hf-sft` có tùy chọn trộn khoảng 10% dòng gọi công cụ tự sinh: đúng dạng JSON mà câu tool_use của bộ chấm yêu cầu, có cả câu không cần công cụ; câu hỏi và số liệu khác bộ chấm; không dùng API trả phí. Notebook train bật tùy chọn này mặc định.
- [ ] 2. Chặn gần trùng giữa dữ liệu train và bộ chấm bằng MinHash của M12; `manifest.json` ghi số dòng bị loại.
- [ ] 3. Chỉ tính loss trên câu trả lời: truyền `assistant_only_loss=True` vào `SFTConfig` (hiện TRL tính loss cả câu hỏi của người dùng). TRL 1.13 tự thay chat template có `{% generation %}` cho Qwen2.5, Qwen3, Qwen3.8; template không hỗ trợ thì báo lỗi tiếng Việt rõ ràng trước khi train, tắt được bằng cấu hình. Test train thật trên CPU với model tí hon dùng chat template Qwen: nhãn phần câu hỏi là -100.
- [ ] 4. `python -m local_ai.check` xanh.

## M16: Notebook bền hơn
Mục tiêu: Colab (chạy từ điện thoại) không chạy tiếp các ô sau khi một lệnh đã lỗi; chạy lại sau khi Colab ngắt thì không phải chấm lại model gốc.
- [x] 1. Mọi lệnh ngoài (python, pip, apt-get, curl, ollama...) trong 2 notebook chạy qua hàm `run` của `local_ai/colab.py` (không qua shell):
  - in output ngay khi có (giữ `\r` cho thanh tiến trình, `PYTHONUNBUFFERED=1` để log train hiện ngay);
  - mã thoát khác 0 thì ném `StepFailed` với lỗi tiếng Việt (bước nào, mã thoát, chạy lại từ đâu), nên ô báo đỏ và Run all dừng lại;
  - notebook không còn dòng `!python`. Chỉ còn `!git clone` ở lần tải đầu (lúc đó chưa có code repo), và nếu tải hỏng thì ô báo lỗi ngay nhờ đoạn kiểm tra thư mục `local_ai`.
  - Không dùng `_exit_code` của Colab, vì không kiểm chứng được Colab có đặt biến này (gói `google-colab` không có trên PyPI).

  Bằng chứng: `RunHelperTests` (tiến trình thật), `NotebookTests.test_no_shell_escapes_except_first_clone`, `test_every_run_call_names_its_own_step`.
- [x] 2. Lệnh eval có `--hub-repo`, `--hub-path`:
  - chấm xong thì đẩy `report.json` và `report.md` (kèm cài đặt: model, `max_new_tokens`, mã băm bộ eval) lên repo riêng tư;
  - chạy lại mà Hub đã có báo cáo cùng cài đặt thì tải về và bỏ qua bước chấm, không nạp model;
  - khác cài đặt, báo cáo thiếu thông tin, hoặc lỗi mạng thì chấm lại; không đẩy được thì chỉ cảnh báo;
  - notebook train dùng cách này cho Bước 7 (chấm trước). Bước 9 (chấm sau) luôn chấm lại vì adapter có thể đã đổi.

  Bằng chứng: `EvalReuseTests`.
- [x] 3. Test: `tests/test_m16_notebook_resilience.py` (11 test).
  - `run` được test với tiến trình thật: output, mã thoát, chữ tiếng Việt, ký tự shell chỉ là chữ, dừng ở ô lỗi.
  - Dùng lại / không dùng lại báo cáo được test với `huggingface_hub` giả.
  - Lệnh của 2 notebook chạy bằng `--dry-run` qua test M10/M11/M13. Các test này được sửa cách đọc lệnh (`run("python ...")` thay cho `!python`), không bỏ kiểm tra nào.
- [x] 4. 226 test chạy qua; compileall và secret-scan sạch.

---

Kế hoạch, tiêu chí và bằng chứng của tuần 1 (M1–M7) và tuần 2 (M8–M14), đều đã xong: `archive/tasks-tuan-1-2.md`.
