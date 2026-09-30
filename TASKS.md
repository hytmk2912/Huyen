# Nhiệm vụ tuần 4 (M23–M27)

Bắt đầu: 2026-09-27, từ `main` tại commit `db0f9d4` (338 test chạy qua, không test nào bị bỏ qua; compileall và secret-scan sạch). Nhánh làm việc: `claude/nhiem-vu-tuan-jixv6i`.

Kế hoạch và bằng chứng các tuần trước: tuần 1–2 (M1–M14) trong `archive/tasks-tuan-1-2.md`, tuần 3 (M15–M22) chép nguyên văn trong `archive/tasks-tuan-3.md`.

Cách làm như tuần 3: mỗi lượt **tối đa 1 mốc** bằng skill `lam-moc`; mốc chỉ **Xong** khi đạt mọi tiêu chí và `python -m local_ai.check` (không `--allow-skip`) xanh; khi đó tự gộp PR. Tiêu chí 5 mốc dưới đây do chủ repo đặt ngày 27/9; mỗi tiêu chí phải có test hoặc lệnh chứng minh.

## Bảng tiến độ tuần 4

| Mốc | Nội dung | Trạng thái | Tiến độ | Bằng chứng |
| --- | --- | --- | --- | --- |
| M23 | Gửi `max_tokens` cho model qua server | **Xong** | 2/2 (100%) | `tests/test_m23_max_tokens.py` (4 test) |
| M24 | Notebook Kaggle | **Xong** | 5/5 (100%) | `tests/test_m24_kaggle.py` (17 test) |
| M25 | Model trung gian 14B abliterated | **Xong** | 4/4 (100%) | `tests/test_m25_model_14b.py` (9 test) |
| M26 | Preset tiếng Việt thứ 2 | **Xong** | 4/4 (100%) | `tests/test_m26_preset_tieng_viet_2.py` (8 test) |
| M27 | Tổng kết tuần 4 | **Chưa làm** | 0/3 (0%) | — |
| **Tổng** | 5 mốc | 4 **Xong** | 4/5 mốc (80%) | `python -m local_ai.check`: 376 test chạy qua, không test nào bị bỏ qua (30/9, sau M26) |

## M23: Gửi max_tokens
Mục tiêu: `max_new_tokens` hiện không giới hạn được độ dài câu trả lời của model qua server (Ollama, llama.cpp, vLLM), vì adapter không gửi `max_tokens` (phát hiện khi làm M18).
- [x] 1. `openai_compatible` gửi `"max_tokens"` = `config.max_new_tokens` trong payload.

  Đã làm: `local_ai/models/openai_compatible.py` thêm `max_tokens` vào payload. README (mục chạy model qua server) ghi rõ.
- [x] 2. Test dùng server giả, kiểm tra payload nhận được có `max_tokens` đúng giá trị, cả giá trị mặc định 512 và giá trị tự đặt. `python -m local_ai.check` xanh.

  Bằng chứng: `tests/test_m23_max_tokens.py` (4 test: mặc định 512; tự đặt 1, 64, 300, 4096; lệnh chấm dùng giá trị trong danh sách model rồi `--max-new-tokens 32`; các mục server trong `platform.json`). Chạy trên code cũ thì cả 4 test lỗi. Test M3 so nguyên payload thêm `"max_tokens": 512`. `python -m local_ai.check`: 342 test, 0 sai, 0 lỗi, 0 bị bỏ qua; KẾT QUẢ: XANH.

## M24: Notebook Kaggle
Làm đúng các bước của `train_colab` trên GPU miễn phí của Kaggle; chưa chạy thử trên Kaggle thật.
- [x] 1. `notebooks/train_kaggle.ipynb` sinh bằng `notebooks/build.py`.

  Bằng chứng: hàm `train_kaggle` trong `notebooks/build.py` (có trong `NOTEBOOKS`); test `test_valid_nbformat_matches_builder_and_has_no_outputs` so file `.ipynb` với nội dung sinh ra.
- [x] 2. Token đọc từ Kaggle Secrets; kết quả ghi ra `/kaggle/working`; train tiếp từ checkpoint trên HF Hub khi phiên bị ngắt.

  Bằng chứng:
  - token: `test_token_is_read_from_kaggle_secrets`, `test_missing_secret_is_explained_in_vietnamese` (chạy ô Bước 4 với `kaggle_secrets` giả), `test_no_colab_secret_or_token_in_notebook`;
  - `/kaggle/working`: `test_code_and_outputs_live_under_kaggle_working` (code ở `/kaggle/working/Huyen`, mọi đường dẫn đầu ra của lệnh nằm dưới đó), `test_result_cell_copies_results_into_kaggle_working` (Bước 13 chép vào `/kaggle/working/ket_qua/<model>`);
  - train tiếp: `test_session_cut_resumes_from_hub_checkpoint` và `test_first_session_trains_from_scratch_and_pushes_checkpoints` (chạy đúng lệnh train của notebook với thư viện và Hub giả);
  - dùng chung repo Hub với Colab (`test_same_commands_as_colab_except_measurement_path`). Số đo lưu ở `so_do/kaggle-<model>.json` nhờ tùy chọn mới `calibrate --hub-path`, không ghi đè số đo Colab (`test_kaggle_measurements_do_not_overwrite_colab_measurements`, `test_invalid_hub_path_is_rejected_in_vietnamese`).
- [x] 3. Notebook hợp lệ theo nbformat, không kèm output, ghim phiên bản thư viện; mọi lệnh chạy được bằng `--dry-run`.

  Bằng chứng: `test_valid_nbformat_matches_builder_and_has_no_outputs`, `test_library_versions_are_pinned_like_colab`, `test_every_code_cell_starts_with_vietnamese_comment`, `test_commands_run_through_run_and_name_their_step`, `test_commands_run_with_dry_run_for_both_models` (8 lệnh × `smoke`, `light`).
- [x] 4. README có mục hướng dẫn chạy trên iPhone.

  Bằng chứng: README mục "Train trên Kaggle miễn phí" (nút Open In Kaggle) và "Hướng dẫn chạy trên iPhone"; test `test_readme_has_kaggle_button_and_iphone_guide`. Test M14 thêm `train_kaggle.ipynb` vào danh sách notebook, và notebook Kaggle cần nút Kaggle thay cho nút Colab (yêu cầu mới của M24).
- [x] 5. `python -m local_ai.check` xanh.

  Bằng chứng: 359 test, 0 sai, 0 lỗi, 0 bị bỏ qua; KẾT QUẢ: XANH.

## M25: Model trung gian 14B abliterated
Model đã chọn: `huihui-ai/Huihui-Qwen3-14B-abliterated-v2`, mục `medium`. Có 2 bản 14B của huihui-ai trên Hugging Face; bản v2 mới hơn, nhiều lượt tải hơn và không bị khóa (bản cũ `Qwen3-14B-abliterated` phải chấp nhận điều khoản trước khi tải).
- [x] 1. `configs/models/platform.json` thêm bản Huihui Qwen3 14B abliterated, ghim revision 40 ký tự lấy từ API Hugging Face (chỉ đọc metadata, không tải trọng số). Không có mạng thì dừng lại và ghi vào "Việc dở".

  Bằng chứng: có mạng. API `https://huggingface.co/api/models/huihui-ai/Huihui-Qwen3-14B-abliterated-v2` ngày 29/9/2026 trả `sha` `3b79629fdd65004d3b9cdf5beb0739e8c7e1becd` và `safetensors.total` 14.768.307.200 (14,77 tỷ). Bản chụp metadata (kèm `config.json` và mã băm `chat_template.jinja` ở đúng revision) nằm ở `tests/fixtures/hf_api/huihui_qwen3_14b_abliterated_v2_2026-09-29.json`. Test: `test_platform_entry_matches_hugging_face_metadata`, `test_chat_template_lets_trl_train_on_answers_only`. Test M19 thêm model 14B vào danh sách model đã ghim revision (yêu cầu mới của M25).
- [x] 2. Thêm `configs/training/colab_14b.json` (QLoRA, fp16).

  Bằng chứng: QLoRA 4bit, fp16, batch 1, tích lũy 16, `max_length` 1024, chỉ tính loss trên câu trả lời. Test `test_colab_14b_is_qlora_fp16_on_medium`.
- [x] 3. Test chứng minh `vram.py` ước tính vừa 16 GB với seq len đã chọn; nếu không vừa thì ghi seq len lớn nhất còn vừa.

  Kết quả: **không vừa**. Ở seq len đã chọn (1024), `vram.py` ước tính 18,6 GB; seq len lớn nhất còn vừa 16 GB là **128 token**, quá ngắn để train có ích. Cần GPU 24 GB (ví dụ L4). `vram.py` thêm tham số `seq_len` (giả định một nửa phần thêm khi train tăng theo số token, chưa đo; ở 2048 token giữ nguyên số cũ), hàm `max_seq_len` và tùy chọn `--seq-len`, `--budget-gb`. Lệnh `estimate` dùng độ dài của cấu hình. Test: `test_chosen_length_does_not_fit_16gb_and_longest_fitting_is_recorded`, `test_seq_len_keeps_calibrated_values_at_2048`, `test_longest_length_for_other_models`, `test_vram_command_prints_length_and_longest_fitting`, `test_estimate_uses_the_length_in_the_config`, `test_readme_explains_the_14b_model_and_its_vram`.
- [x] 4. `python -m local_ai.check` xanh.

  Bằng chứng: 368 test, 0 sai, 0 lỗi, 0 bị bỏ qua; KẾT QUẢ: XANH.

## M26: Preset tiếng Việt thứ 2
Preset mới: `vietnamese_aya` (`configs/datasets/presets/vietnamese_aya.json`), lấy câu hỏi và câu trả lời tiếng Việt do người viết mới hoàn toàn trong `CohereLabs/aya_dataset`.
- [x] 1. Chỉ chọn dataset có giấy phép cho phép dùng.

  Bằng chứng: card ghi Apache-2.0, "can be used for any purpose, whether academic or commercial"; dữ liệu do người tình nguyện của Aya Open Science Initiative viết, không bị khóa. Preset chỉ lấy dòng `language_code` `vie` và `annotation_type` `original-annotations` (4.853 dòng ở commit `f9ea0458…`), bỏ `re-annotations` vì phần gốc đến từ dataset khác. Các ứng viên bị loại: bản dịch từ dữ liệu sinh bằng ChatGPT hoặc Alpaca (cấm thương mại), dataset không ghi nguồn gốc, dataset bị khóa. Test `test_preset_points_to_a_pinned_dataset_with_a_license_that_allows_use`.
- [x] 2. Ghi giấy phép vào `docs/GIAY_PHEP_DATASET.md`.

  Bằng chứng: dòng `vietnamese_aya` trong bảng tóm tắt và mục riêng (giấy phép, nguồn gốc, cảnh báo của card, cách ghi nguồn). Test `test_license_is_written_in_the_license_doc`.
- [x] 3. Có fixture và test giống các preset cũ.

  Bằng chứng: fixture `tests/fixtures/presets/vietnamese_aya.jsonl` (6 dòng: tiếng Anh, `re-annotations`, dòng thiếu câu trả lời, và 3 dòng hợp lệ, một dòng lấy thật từ Aya). Test `tests/test_m26_preset_tieng_viet_2.py` (8 test): map cột, chỉ giữ dòng tiếng Việt do người viết, trộn thay preset `vietnamese`, lệnh `--dry-run`, bộ lọc `where` và `scan_limit` mới trong `hub.py`. Test M4 thêm preset mới vào danh sách preset (yêu cầu mới của M26). `where` ghi dạng cặp `[cột, giá trị]`, vì test M1 coi mọi khóa trong cấu hình là khóa code phải đọc (như tham số Ollama ở M18). Đã đọc thật 20 dòng bằng streaming (30/9): 20/20 dòng tiếng Việt, không dòng nào bị loại.
- [x] 4. `python -m local_ai.check` xanh.

  Bằng chứng: 376 test, 0 sai, 0 lỗi, 0 bị bỏ qua; KẾT QUẢ: XANH.

## M27: Tổng kết tuần 4
- [ ] 1. README, `docs/ARCHITECTURE.md`, `memory.md` khớp thực tế.
- [ ] 2. Có đề xuất tuần 5.
- [ ] 3. `python -m local_ai.check` xanh.
