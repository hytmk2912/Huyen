# Nhiệm vụ 1 tuần

Bắt đầu: 2026-09-24, từ `main` tại commit `60e8541` (28 test chạy qua, 1 bỏ qua; compileall và secret-scan sạch).

Cách làm: mỗi lượt làm **tối đa 1 mốc**, theo thứ tự M1 → M7, bằng skill `lam-moc` (gõ `/lam-moc` trong Claude Code hoặc nói "làm mốc tiếp theo"). Mốc đang làm, việc dở và lỗi gặp được ghi trong `memory.md`.

## Cách chấm
- Mỗi mốc có danh sách tiêu chí `[ ]`. Chỉ đánh `[x]` khi tiêu chí đã đạt và có bằng chứng (tên test hoặc lệnh kiểm tra).
- % của mốc = số tiêu chí đạt ÷ tổng số tiêu chí của mốc. Tổng % = trung bình cộng 7 mốc.
- Mốc chỉ được ghi **Xong** khi đạt đủ 100% tiêu chí **và** cả ba lệnh kiểm tra đều xanh:
  `python -m unittest discover -s tests -v`, `python -m compileall -q local_ai`, `python -m local_ai.data secret-scan`.
- Test không được tải model hay dataset từ mạng.

## Bảng tiến độ

| Mốc | Nội dung | Ngày gợi ý | Trạng thái | Tiến độ | Bằng chứng |
| --- | --- | --- | --- | --- | --- |
| M1 | Agent không sập khi công cụ lỗi; dọn repo | Ngày 1 | **Xong** | 9/9 (100%) | `tests/test_m1_agent_cleanup.py` (11 test) |
| M2 | Model ảnh+chữ, nén 4-bit (QLoRA), ước tính VRAM | Ngày 2 | **Xong** | 8/8 (100%) | `tests/test_m2_models.py` (20 test) |
| M3 | Adapter gọi server local kiểu OpenAI | Ngày 3 | **Xong** | 7/7 (100%) | `tests/test_m3_local_server.py` (17 test) |
| M4 | Preset dataset Hugging Face | Ngày 4 | **Xong** | 9/9 (100%) | `tests/test_m4_presets.py` (14 test) + chạy thật 20 dòng/preset |
| M5 | Đánh giá (eval) mở rộng | Ngày 5 | **Xong** | 8/8 (100%) | `tests/test_m5_eval.py` (20 test) |
| M6 | Test chạy thật trên CPU với model tí hon | Ngày 6 | **Xong** | 7/7 (100%) | `tests/test_m6_cpu_pipeline.py` (3 test, chạy thật khoảng 6 giây) |
| M7 | Tổng kết | Ngày 7 | Chưa làm | 0/5 (0%) | — |
| **Tổng** | | | | **86%** (600% ÷ 7) | |

## Ngoài phạm vi tuần này
Không làm: pretrain/corpus quy mô lớn, huấn luyện phân tán, xuất GGUF, gọi API trả phí, tải trọng số model hay dataset lớn. Việc phát sinh ngoài 7 mốc: ghi vào "Việc dở" trong `memory.md` và hỏi chủ repo trước.

---

## M1: Agent không sập khi công cụ lỗi; dọn repo

**Hiện trạng** (đã chạy thử trên `main` ngày 24/9):
- `calculator("1/0")` ném `ZeroDivisionError` và `calculator("2 +")` ném `SyntaxError` ra ngoài, làm sập cả agent, vì `ToolRegistry.execute` chỉ bắt `TypeError` và `ValueError`.
- `_parse_json_object` trả về `{}` khi JSON có chữ thừa, nằm trong khối ```` ```json ````, hoặc đứng sau `<think>`.

**Tiêu chí xong**

Agent:
- [x] 1. Công cụ ném bất kỳ ngoại lệ nào (kể cả `ZeroDivisionError`, `SyntaxError`) thì `ToolRegistry.execute` trả kết quả thất bại kèm thông báo lỗi, không ném ra ngoài. Có test với `1/0` và `2 +`. Bằng chứng: `tests/test_m1_agent_cleanup.py` `ToolErrorTests.test_division_by_zero_and_syntax_error_do_not_raise`.
- [x] 2. Agent chạy hết vòng, không sập khi công cụ lỗi: lỗi được ghi vào trace và model được thử lại. Có test: model giả gọi `1/0`, sau đó sửa thành phép tính đúng, và agent hoàn thành. Bằng chứng: `tests/test_m1_agent_cleanup.py` `ToolErrorTests.test_agent_survives_tool_error_and_retries`.
- [x] 3. Tách được JSON từ câu trả lời model trong các trường hợp: JSON thuần; có chữ thừa trước hoặc sau; nằm trong khối ```` ```json ```` hoặc ```` ``` ````; đứng sau `<think>...</think>` (bỏ qua JSON nằm trong `<think>`); chuỗi trong JSON chứa dấu `{` `}`. Mỗi trường hợp có một test. Bằng chứng: `tests/test_m1_agent_cleanup.py` `JsonExtractionTests (test_each_format, test_braces_inside_strings_and_nested_objects, test_no_json_or_only_json_inside_think)`.
- [x] 4. Agent hoàn thành nhiệm vụ khi model giả trả lời theo các kiểu ở tiêu chí 3. Bằng chứng: `tests/test_m1_agent_cleanup.py` `JsonExtractionTests.test_agent_completes_with_messy_model_output`.

Dọn repo:
- [x] 5. Cất phần corpus 10T vào `archive/corpus-10t/`: `local_ai/data/corpus.py`, `configs/datasets/corpus_10t.json`, `configs/datasets/smoke_real.json`, các lệnh CLI corpus và test đi kèm. Code trong `local_ai/` không còn import phần này, và test đã cất không còn chạy trong `tests/`. Bằng chứng: `tests/test_m1_agent_cleanup.py` `CleanupTests.test_corpus_is_archived_and_not_imported`.
- [x] 6. Lệnh `python -m local_ai.data secret-scan` vẫn chạy: hàm quét được tách ra một file riêng trong `local_ai/` và có test. Bằng chứng: `tests/test_m1_agent_cleanup.py` `CleanupTests.test_secret_scan_command_still_works`.
- [x] 7. Không còn khóa cấu hình thừa: mọi khóa trong `configs/**/*.json` (trừ `_comment`) đều được code đọc, có test tự dò. Khóa thừa hiện chỉ nằm trong `corpus_10t.json` và `smoke_real.json`. Bằng chứng: `tests/test_m1_agent_cleanup.py` `CleanupTests.test_every_config_key_is_read_by_code`.
- [x] 8. Không còn link `huyenb2404-ops` (hiện nằm ở `data/raw/seed_examples.jsonl` và `data/eval/seed_eval.jsonl`); đổi thành `https://github.com/hytmk2912/Huyen`. Bằng chứng: `tests/test_m1_agent_cleanup.py` `CleanupTests.test_old_repo_link_is_gone`.
- [x] 9. README khớp hướng fine-tune: mô tả luồng dataset Hugging Face → sft.jsonl → LoRA/QLoRA → đánh giá, bỏ mục tiêu 10T token, phần lộ trình trỏ tới `TASKS.md`. Bằng chứng: `tests/test_m1_agent_cleanup.py` `CleanupTests.test_readme_follows_finetune_direction`.

---

## M2: Model ảnh+chữ, nén 4-bit (QLoRA), ước tính VRAM

**Hiện trạng trước M2:**
- Hugging Face gắn nhãn `image-text-to-text` (kiến trúc `qwen3_5`, lớp `AutoModelForMultimodalLM`, 27,78 tỷ tham số) cho model chính `huihui-ai/Huihui-Qwen3.8-27B-abliterated`, nhưng adapter chỉ nạp bằng `AutoModelForCausalLM`.
- Trường `quantization` trong `ModelConfig` có khai báo nhưng không code nào đọc. Chưa có QLoRA.

**Tiêu chí xong** (cập nhật ngày 24/9 theo yêu cầu mới của chủ repo: khóa `kind`/`params_b`, lớp `AutoModelForMultimodalLM`, nén `4bit`/`8bit`, tự chuyển fp16)
- [x] 1. Mỗi model trong `configs/models/platform.json` có `kind` (`text` hoặc `multimodal`) và `params_b` (số tỷ tham số); model chính là `multimodal`, 27,78 tỷ, theo thẻ trên Hugging Face. Giá trị sai bị từ chối. Bằng chứng: `tests/test_m2_models.py` `ModelKindConfigTests` (2 test).
- [x] 2. Adapter chọn đúng lớp nạp: `AutoProcessor` + `AutoModelForMultimodalLM` cho model multimodal (bản transformers không có lớp này thì dùng `AutoModelForImageTextToText`, không có cả hai thì báo lỗi rõ); `AutoTokenizer` + `AutoModelForCausalLM` cho model text. Test dùng module `transformers` giả. Bằng chứng: `ModelLoadingTests` (3 test).
- [x] 3. Gửi được tin nhắn gồm ảnh (đường dẫn file cục bộ) và chữ tới model multimodal. Gửi ảnh tới model text, hoặc ảnh không tồn tại, thì báo lỗi rõ bằng tiếng Việt. Bằng chứng: `ImageMessageTests` (2 test).
- [x] 4. `quantization: "4bit"` (NF4, double quant) hoặc `"8bit"` trong cấu hình model thì model được nạp nén bằng bitsandbytes; thiếu bitsandbytes thì báo lỗi rõ. Bằng chứng: `QuantizationTests` (2 test).
- [x] 5. `finetune` hỗ trợ QLoRA: `method: "lora"` + `quantization: "4bit"` → nạp model 4bit, `prepare_model_for_kbit_training` rồi gắn LoRA. Full fine-tune trên model nén bị từ chối; thiếu GPU hoặc bitsandbytes thì trả `"skipped"`. Có cấu hình mẫu `configs/training/qlora_primary.json` cho model chính. (Không dùng `method: "qlora"` vì test cũ `test_config_resolves_base_model_and_overrides` yêu cầu method đó bị từ chối.) Bằng chứng: `QloraFinetuneTests` (5 test).
- [x] 6. GPU không có bf16 (ví dụ T4) thì tự chuyển sang fp16 và in cảnh báo, cả khi nạp model lẫn khi train. Bằng chứng: `DtypeFallbackTests` (3 test), `QloraFinetuneTests.test_t4_trains_in_fp16`.
- [x] 7. Lệnh `python -m local_ai.models.vram` in bảng cho **mọi** model trong `configs/models/platform.json`: số tham số, trọng số ở bf16/8bit/4bit, train LoRA/QLoRA. Không cần mạng, không import torch/transformers, ghi rõ là ước lượng. Bằng chứng: `VramEstimateTests.test_command_prints_every_model_without_loading`, `test_model_without_params_is_marked`.
- [x] 8. Số liệu ước tính cho model 27B: bf16 = 54 GB, 4bit trong khoảng 14–16 GB, train QLoRA dưới 24 GB. Bằng chứng: `VramEstimateTests.test_numbers_for_27b`.

---

## M3: Adapter gọi server local kiểu OpenAI (Ollama, llama.cpp, vLLM)

**Tiêu chí xong**
- [x] 1. Adapter gửi `POST {base_url}/chat/completions` theo chuẩn OpenAI và trả về `choices[0].message.content`. Dùng được với Ollama (`http://localhost:11434/v1`), llama.cpp server (`http://localhost:8080/v1`) và vLLM (`http://localhost:8000/v1`). Chỉ dùng `urllib` của Python, không thêm thư viện `openai`. Bằng chứng: `tests/test_m3_local_server.py` `RequestTests.test_success_sends_openai_payload_and_returns_content`.
- [x] 2. Khai báo trong `configs/models/platform.json`: `backend: "openai_compatible"`, `base_url`, `source` (tên model trên server), `timeout_s`. Có 2 mục mẫu: `ollama` và `llamacpp`. Khóa truy cập, nếu server yêu cầu, chỉ đọc từ biến môi trường có tên ghi ở `api_key_env`. Bằng chứng: `RequestTests.test_api_key_is_read_only_from_environment`, `ConfigRoutingTests.test_platform_declares_ollama_and_llamacpp`, `LocalAddressTests.test_config_is_validated`.
- [x] 3. Chỉ cho phép địa chỉ máy này hoặc mạng nội bộ. Địa chỉ Internet bị từ chối trước khi gửi, để không vô tình gọi API trả phí. Adapter không dùng proxy và không đi theo redirect. Bằng chứng: `LocalAddressTests` (3 test), `ErrorTests.test_bad_response_and_redirect`.
- [x] 4. Lỗi mạng, hết thời gian chờ, mã lỗi HTTP hay trả lời sai chuẩn đều được báo rõ ràng bằng tiếng Việt. Agent không sập khi server lỗi: agent dừng lại và trả kết quả chưa hoàn thành, kèm thông báo lỗi. Bằng chứng: `ErrorTests` (4 test), `ConfigRoutingTests.test_agent_does_not_crash_when_server_fails`.
- [x] 5. Router và agent dùng được adapter này qua cấu hình (`create_adapter`, `ModelRouter.from_configs`). Demo chạy được qua adapter: `python -m local_ai.demo --model <tên>`. Fine-tune từ chối model chạy qua server. Bằng chứng: `ConfigRoutingTests.test_agent_runs_through_router_built_from_config`, `DemoTests` (2 test), `ConfigRoutingTests.test_finetune_rejects_server_models`.
- [x] 6. Có test dựng server giả bằng `http.server` trên 127.0.0.1 cho các trường hợp: trả lời đúng, trả mã 500, chậm quá thời gian chờ, và server chưa chạy. Test không cần Ollama hay llama.cpp thật. Bằng chứng: `tests/test_m3_local_server.py` (lớp `FakeServer`, 17 test).
- [x] 7. README có hướng dẫn chạy với Ollama, llama.cpp và vLLM: lệnh khởi động server, cấu hình mẫu và lệnh demo. Bằng chứng: `ReadmeTests.test_readme_explains_ollama_llamacpp_and_vllm`.

Chưa làm (ngoài tiêu chí): gửi ảnh qua server local. Hiện adapter báo lỗi rõ ràng khi có ảnh.

---

## M4: Preset dataset Hugging Face (code, reasoning, tiếng Việt)

**Tiêu chí xong** (thêm tiêu chí 7–9 ngày 24/9 theo yêu cầu của chủ repo: trộn nhiều preset theo tỉ lệ, lệnh liệt kê preset, đọc thử 20 dòng thật)
- [x] 1. Có 3 file preset trong `configs/datasets/presets/`: `code` (`bigcode/self-oss-instruct-sc2-exec-filter-50k`), `reasoning` (`open-r1/OpenR1-Math-220k`), `vietnamese` (`5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`). Mỗi file trỏ tới dataset có thật, `revision` là commit cố định. Bằng chứng: `tests/test_m4_presets.py` `PresetFileTests.test_three_presets_point_to_real_datasets`.
- [x] 2. Mỗi preset ghi giấy phép theo trang dataset, kèm trạng thái `"cần kiểm tra lại trên dataset card"`. README nhắc đọc lại giấy phép trước khi dùng. Bằng chứng: `PresetFileTests.test_three_presets_point_to_real_datasets`, `ReadmeTests`.
- [x] 3. Mỗi preset bật `streaming: true`, `limit` ≤ 1000; loader chỉ đọc đúng `limit` dòng. Bằng chứng: `PresetFileTests.test_presets_stream_with_small_limit`, `test_loader_reads_only_limit_rows`.
- [x] 4. Chạy được `python -m local_ai.data hf-sft --config configs/datasets/presets/<tên>.json`. Bằng chứng: `CommandTests.test_hf_sft_accepts_config_or_several_presets`; đã chạy thật cả 3 preset với `--limit 20` (xem `memory.md`).
- [x] 5. `sft.jsonl` xuất đúng: hội thoại nhiều lượt giữ nguyên (kể cả system, hiểu cả dạng ShareGPT from/value); reasoning vào khối `<think>`; context đặt trước câu hỏi. Hội thoại sai dạng (ví dụ kết thúc bằng câu hỏi) bị loại. Bằng chứng: `PresetMappingTests` (`test_multi_turn_conversation_is_kept_with_system`, `test_reasoning_is_kept_in_think_block`, `test_context_is_put_before_question`).
- [x] 6. Test dùng loader giả (không mạng) cho từng preset: ánh xạ cột đúng, bản ghi hợp lệ, loader nhận đúng `name`/`subset`/`revision` và `streaming=True`. Bằng chứng: `PresetMappingTests.test_each_preset_maps_columns_and_calls_loader_with_streaming`, `test_code_rows`.
- [x] 7. `hf-sft` nhận nhiều preset kèm tỉ lệ (`--preset code:0.4 --preset reasoning:0.3 ...`, `--total`, `--limit`), xuất chung một `sft.jsonl`, `manifest.json` ghi tỉ lệ và số dòng từng nguồn. Bằng chứng: `MixTests` (3 test), `CommandTests.test_hf_sft_accepts_config_or_several_presets`.
- [x] 8. Lệnh `python -m local_ai.data list-presets` liệt kê preset: dataset, commit, domain, ngôn ngữ, số dòng tối đa, giấy phép và trạng thái. Bằng chứng: `CommandTests.test_list_presets_command`.
- [x] 9. Đọc thử thật 20 dòng mỗi preset qua mạng (không nằm trong test), ghi kết quả vào `memory.md`.

---

## M5: Đánh giá (eval) mở rộng

**Hiện trạng trước M5:** `local_ai/evaluation/benchmarks.py` chỉ chấm khớp đúng từng chữ; bước chặn rò rỉ eval chỉ so ID.

**Tiêu chí xong** (thêm tiêu chí 8 ngày 24/9 theo yêu cầu của chủ repo: cách chấm `exact`, lệnh eval theo tên model, báo cáo JSON + Markdown trong `.runs/`)
- [x] 1. Chấm `contains`: không phân biệt hoa thường, chuẩn hóa khoảng trắng và Unicode tiếng Việt (NFC). `exact` chuẩn hóa như vậy. Khối `<think>` bị bỏ trước khi chấm. Bằng chứng: `tests/test_m5_eval.py` `ScoringTests.test_contains_normalizes_case_space_and_vietnamese_unicode`, `test_exact`.
- [x] 2. Chấm `regex`. Bằng chứng: `ScoringTests.test_regex`.
- [x] 3. Chấm `python_tests`: chạy code của model cùng unit test trong `PythonSandbox`, có `timeout_s`. Code sai cú pháp, lỗi khi chạy hoặc lặp vô hạn đều tính là không đạt, chương trình không sập. Bằng chứng: `ScoringTests.test_python_tests_pass_and_fail`, `test_python_syntax_runtime_error_and_infinite_loop_do_not_crash`.
- [x] 4. `data/eval/eval_v1.jsonl` có 30 câu: 15 tiếng Việt, 15 tiếng Anh; 8 `code` (có unit test), 14 `reasoning`, 8 `tool_use`. Mỗi câu ghi `language`, `group`, `scoring` và có `reference` (đáp án mẫu); mọi đáp án mẫu đều đạt. Bằng chứng: `CaseFileTests` (3 test).
- [x] 5. Báo cáo theo nhóm và theo ngôn ngữ, liệt kê câu không đạt kèm lý do. Bằng chứng: `RunTests.test_report_by_group_and_language_lists_failures`.
- [x] 6. Mỗi cách chấm có test cả trường hợp đạt và không đạt, kể cả code lặp vô hạn. Bằng chứng: `ScoringTests` (6 test).
- [x] 7. Không câu eval nào lọt vào dữ liệu train: kiểm tra theo id, theo nội dung (content_hash) và theo câu hỏi đã chuẩn hóa (kể cả các lượt user trong hội thoại). Bước build dừng khi có trùng; mọi cấu hình dataset chặn cả `eval_v1.jsonl`. Lệnh eval có `--train-data` để từ chối chạy khi dữ liệu train chứa câu eval. Bằng chứng: `OverlapTests` (3 test), `CommandTests.test_training_data_overlapping_eval_is_blocked`. Test cũ `test_data_factory.test_build_versions_and_prevents_eval_leakage` được đổi nội dung bản ghi eval, vì trước đó nó dựa vào đúng lỗi này (train và eval cùng nội dung, chỉ khác id).
- [x] 8. Lệnh `python -m local_ai.evaluation --model <tên>` (hoặc `--scripted`: chạy bằng ScriptedModelAdapter trả đáp án mẫu) ghi `report.json` và `report.md` vào `.runs/eval/<model>-<thời điểm>/`. Model lỗi thì dừng, báo `status: "error"`, không sập. README có hướng dẫn. Bằng chứng: `RunTests` (4 test), `CommandTests` (3 test), `ReadmeTests`.

---

## M6: Test chạy thật trên CPU với model tí hon

**Tiêu chí xong** (thêm tiêu chí 7 ngày 24/9 theo yêu cầu của chủ repo: nạp lại adapter rồi eval)
- [x] 1. Test tự tạo tokenizer (BPE byte-level, có `chat_template`) và model tí hon khởi tạo ngẫu nhiên (Llama 2 lớp, hidden 64), không tải model, tokenizer hay dataset nào. Test chạy với `HF_HUB_OFFLINE=1`; đã chạy lại khi chặn hẳn mạng (proxy sai) và vẫn xanh. Bằng chứng: `tests/test_m6_cpu_pipeline.py` `CpuPipelineTests.test_fixture_to_lora_to_reload_to_eval`.
- [x] 2. Phần dữ liệu: từ fixture của 3 preset trong repo, qua bước build (`prepare_hf_mix`), ra `sft.jsonl` (8 dòng). Bằng chứng: như trên.
- [x] 3. Train LoRA đúng 2 bước trên CPU qua `local_ai.training.finetune` (`require_gpu: false`, khóa mới `max_steps`) và tạo ra adapter (`adapter_config.json`, `adapter_model.safetensors`). Bằng chứng: như trên (`result["steps"] == 2`).
- [x] 4. Phần eval: chạy bộ eval 30 câu với model vừa train và ghi `report.json` và `report.md`. Điểm thấp là bình thường. Bằng chứng: như trên.
- [x] 5. Toàn bộ test M6 chạy khoảng 6 giây trên CPU (dưới 3 phút). Máy thiếu torch, transformers, peft, trl hoặc datasets thì test tự bỏ qua (đã kiểm tra bằng venv trống: `OK (skipped=1)`).
- [x] 6. Đã chạy thật trên máy có đủ thư viện (torch 2.14.0+cpu, transformers 5.17.0, trl 1.13.0, peft 0.21.0, datasets 5.0.1); thời gian chạy và loss ghi trong `memory.md`.
- [x] 7. Nạp lại được model gốc cùng adapter qua khóa mới `adapter_path` trong danh sách model (model thành `PeftModelForCausalLM`) để eval. Lệnh `python -m local_ai.training.finetune` và `python -m local_ai.evaluation --model <tên có adapter_path>` chạy được trên model tí hon. Bằng chứng: `CpuPipelineTests`, `AdapterPathTests` (2 test, dùng module giả nên chạy được cả khi thiếu thư viện).

---

## M7: Tổng kết

**Tiêu chí xong**
- [ ] 1. README cập nhật đúng trạng thái thật: cài đặt, luồng làm việc và lệnh cho từng bước. Mọi lệnh trong README đều đã chạy thử được.
- [ ] 2. Có lộ trình sau tuần 1: vài việc tiếp theo, ghi rõ là đề xuất, chưa làm.
- [ ] 3. Chấm lại % từng mốc trong bảng tiến độ theo tiêu chí, kèm bằng chứng (tên test, commit), và ghi tổng %.
- [ ] 4. `memory.md` có tổng kết tuần: việc đã xong, việc còn dở, lỗi còn tồn.
- [ ] 5. Cả ba lệnh kiểm tra (test, compileall, secret-scan) đều xanh.
