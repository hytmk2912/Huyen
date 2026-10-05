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
| M27 | Tổng kết tuần 4 | **Xong** | 3/3 (100%) | `tests/test_m27_tong_ket_tuan_4.py` (7 test) |
| **Tổng** | 5 mốc | 5 **Xong** | 5/5 mốc (100%) | `python -m local_ai.check`: 383 test chạy qua, không test nào bị bỏ qua (1/10, sau M27) |

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
- [x] 1. README, `docs/ARCHITECTURE.md`, `memory.md` khớp thực tế.

  Bằng chứng:
  - README: mục trạng thái cập nhật 1/10 (thêm M23–M26 và phần chưa chạy thật); nút Kaggle ở đầu trang; lộ trình đánh dấu M23–M27 xong, lộ trình tuần 3 giữ nguyên bên dưới; chỗ nhắc kế hoạch trỏ tới `TASKS.md` tuần 4 và `archive/tasks-tuan-3.md`.
  - `docs/ARCHITECTURE.md`: phạm vi có Kaggle, adapter server gửi `max_tokens`, test M23, M25, M26, mục giới hạn cập nhật 1/10.
  - `memory.md`: XONG TUẦN 4, checklist đủ 5 mốc, nhật ký rút gọn (chi tiết ở `archive/memory-tuan-4.md`), 4.316 ký tự; mục "Việc chủ repo tự làm" giữ nguyên.
  - Test: `test_every_milestone_is_done_with_all_criteria`, `test_test_counts_in_table_match_test_files`, `test_roadmap_marks_week_four_done_and_status_is_updated`, `test_readme_numbers_match_code`, `test_memory_says_the_week_is_done`, `test_architecture_lists_week_four_parts_and_what_is_still_unverified`.
  - Test M21 trước đòi mục trạng thái ghi đúng "Cập nhật 27/9/2026"; nay chỉ đòi ngày cập nhật không sớm hơn 27/9/2026 (yêu cầu mới của M27).
- [x] 2. Có đề xuất tuần 5.

  Bằng chứng: 4 đề xuất giống nhau ở README (mục "Đề xuất tuần 5") và `memory.md`. Test `test_week_five_proposals_match_between_readme_and_memory`.
- [x] 3. `python -m local_ai.check` xanh.

  Bằng chứng: 383 test, 0 sai, 0 lỗi, 0 bị bỏ qua; KẾT QUẢ: XANH.

## Hardening bảo mật (chủ repo yêu cầu 5/10, không phải mốc)
Nhánh `security/hardening` (tạo từ `main` tại `c2610bc`). Repo nhận diện là **Huyen** (fine-tune LLM, tải từ Hugging Face) nên làm phần 6B, bỏ phần 6A (ví ETH).

- [x] 1. Quét khóa bị lộ bằng gitleaks 8.28.0 `detect --source . --log-opts="--all"` (106 commit mọi nhánh): 3 phát hiện, đều báo nhầm (`tokenizer_revision` là mã commit, `AutoTokenizer`, tên file test), đã thêm ngoại lệ hẹp trong `.gitleaks.toml` → **0 leak**. Grep thêm (hex 64, `hf_`, `sk-`, `ghp_`, `github_pat_`, `PRIVATE_KEY`, dòng mở đầu khóa riêng dạng PEM, mật khẩu): hex 64 đều là sha256 checksum. **Secret thật đã lộ:** mật khẩu rsync làm giá trị mặc định của `SYNC_PASS` trong `cpu_hub.sh`, `gpu_8h.sh` (commit `e6bc723`, 7 commit chứa 2 file). Không tự xoá lịch sử; lệnh `git filter-repo` chờ chủ repo duyệt (`SECURITY.md` mục 5).
- [x] 2. `.gitignore` thêm `*.key`, `*.pem`, `keystore/`, `wallet*.json`, `*.pkl`, `checkpoints/`, `outputs/`, giữ `!.env.example`; `.env.example` liệt kê biến, giá trị trống. Bằng chứng: `SecretModuleTests.test_gitignore_blocks_secret_and_artifact_files`, `test_env_example_lists_every_known_secret_with_empty_values`.
- [x] 3. Module secret duy nhất `local_ai/config/secrets.py` (`require_secret`, `optional_secret`, `hf_token`, `redact`); thiếu biến bắt buộc → `MissingSecretError` (SystemExit, chỉ báo tên biến), không có giá trị mặc định. Gateway, adapter server local, `data/hub.py` đọc qua module này. Bằng chứng: `SecretModuleTests` (8 test), `HfTokenNotPersistedTests` (4 test: token không vào `manifest.json`, kế hoạch train, lỗi gateway).
- [x] 4. pre-commit (`gitleaks`, `detect-private-key`, `check-added-large-files`, ghim SHA): `pre-commit run --all-files` → 3/3 Passed. `.github/workflows/security.yml`: gitleaks (tải bản chính thức, kiểm tra sha256) + pip-audit trên push và pull_request.
- [x] 5. GitHub Actions: `permissions: contents: read` cấp top-level, action ghim SHA, không `pull_request_target`, không `${{ github.event.* }}` trong `run`; `.github/dependabot.yml` (pip, github-actions, hằng tuần). Bằng chứng: `RepositorySecurityFilesTests` (4 test) và grep trong báo cáo PR.
- [x] 6. Thư viện: `requirements.txt`, `requirements-gpu.txt`, `requirements-dev.txt` ghim `==`; `requirements.lock` (76 gói, kèm hash, sinh bằng `uv pip compile --generate-hashes`). `pip-audit` cả hai bộ: "No known vulnerabilities found".
- [x] 7. Nạp model an toàn (6B.1): `trust_remote_code=False`, `use_safetensors=True` ở mọi chỗ nạp model; chỉ có `.bin`/`.pkl` → từ chối, cần cờ `--allow-pickle-weights` (finetune, export, evaluation). 5 model Hugging Face trong danh sách đều có safetensors (API ngày 5/10, `tests/fixtures/hf_api/safetensors_2026-10-05.json`). Bằng chứng: `SafeModelLoadingTests` (7 test, gồm quét AST mọi `from_pretrained`, cấm `pickle.load`/`torch.load` thiếu `weights_only=True`).
- [x] 8. Ghim revision (6B.2): đọc danh sách model mà model/tokenizer Hugging Face chưa ghim mã commit 40 ký tự → từ chối; tải dataset thật chưa ghim → từ chối; `hf_sft.json` mẫu bỏ `main`. Bằng chứng: `PinnedRevisionTests` (4 test).
- [x] 9. HF token (6B.3) chỉ từ env `HF_TOKEN`, không vào output/log: `HfTokenNotPersistedTests`.
- [x] 10. Path traversal (6B.4): `safe_output_dir` (chỉ `data/`, thư mục tạm, `LOCAL_AI_DATA_DIR`; chặn `../`, symlink), tên preset chỉ chữ/số/`_`/`-`. Bằng chứng: `PathTraversalTests` (5 test).
- [x] 11. `SECURITY.md`: secret và nơi lưu, checklist việc chủ repo làm trên GitHub, quy trình khi lộ secret, secret đã lộ.
- [x] 12. Kiểm tra: `python -m local_ai.check` → 415 test, 0 sai, 0 lỗi, 0 bị bỏ qua; compileall sạch; secret-scan sạch; `KẾT QUẢ: XANH`.
- [x] 13. Mở pull request `security/hardening` → `main` (bản nháp): PR #29. Chờ chủ repo duyệt lệnh `git filter-repo` và làm checklist trên GitHub.
