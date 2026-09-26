> Chép nguyên văn từ `TASKS.md` ngày 26/9/2026 (lượt rà soát tuần 3), để `TASKS.md` chỉ còn tuần 3. Tuần 1 và tuần 2 đều đã xong.

# Nhiệm vụ tuần 2 (M8–M14)

Bắt đầu: 2026-09-24, từ `main` tại commit `09c93f3` (119 test chạy qua; compileall và secret-scan sạch). Nhánh làm việc: `claude/nhiem-vu-tuan-2`.

Cách làm và cách chấm giống tuần 1 (xem phần tuần 1 bên dưới): mỗi lượt làm **tối đa 1 mốc** bằng skill `lam-moc`. Làm các mốc tồn của tuần 1 trước (hiện không có), rồi M8 → M14. Mốc chỉ **Xong** khi đạt mọi tiêu chí và cả ba lệnh kiểm tra đều xanh.

Quy tắc riêng tuần 2:
- Notebook lưu không kèm output, ghim phiên bản thư viện, mỗi ô có chú thích tiếng Việt dễ hiểu (chủ repo dùng điện thoại).
- Notebook không chạy được trong môi trường phát triển: chỉ kiểm tra hợp lệ và chạy thử lệnh của nó bằng `--dry-run`.
- Token chỉ đọc từ biến môi trường hoặc Colab Secrets, không ghi vào repo hay notebook.

## Bảng tiến độ tuần 2

| Mốc | Nội dung | Trạng thái | Tiến độ | Bằng chứng |
| --- | --- | --- | --- | --- |
| M8 | Gộp repo Agent, phần 1: đưa code runtime vào | **Xong** | 4/4 (100%) | `tests/test_m8_runtime.py` (14 test) |
| M9 | Gộp repo Agent, phần 2: tool chạy lệnh an toàn | **Xong** | 5/5 (100%) | `tests/test_m9_terminal.py` (11 test) |
| M10 | Notebook train trên Colab free (0.5B) | **Xong** | 5/5 (100%) | `tests/test_m10_colab.py` (16 test) |
| M11 | Colab cho Qwen3-4B + hướng dẫn iPhone | **Xong** | 4/4 (100%) | `tests/test_m11_colab_light.py` (11 test) |
| M12 | Chất lượng dữ liệu | **Xong** | 3/3 (100%) | `tests/test_m12_quality.py` (18 test) |
| M13 | Agent chạy model thật trên Colab | **Xong** | 3/3 (100%) | `tests/test_m13_agent_colab.py` (11 test) |
| M14 | Tổng kết tuần 2 | **Xong** | 4/4 (100%) | `tests/test_m14_summary.py` (6 test) |
| **Tổng** | 7 mốc tuần 2 | **Xong** | 7/7 (100%) | 206 test chạy qua; compileall và secret-scan sạch |

## M8: Gộp repo Agent, phần 1: đưa code vào
- [x] 1. Lấy được repo Agent. `hytmk2912/Agent` không truy cập được từ phiên (có thể đang riêng tư); `huyenytmk2912/agent` công khai, đã `git clone`, commit `78a3e25`.
- [x] 2. `docs/GOP_AGENT.md`: phần đưa vào (bảng đối chiếu), phần để M9, phần bỏ (FastAPI, uvicorn, pydantic, python-dotenv), 7 rủi ro bảo mật (nghiêm trọng nhất: chèn lệnh qua `shell=True`), commit gốc đã lấy. Code gốc cất nguyên văn ở `archive/agent-goc/`.
- [x] 3. Runtime nằm trong `local_ai/runtime/` (`executor.py`, `jobs.py`, `auth.py`), chỉ dùng thư viện chuẩn, bịt ngay lỗ hổng chèn lệnh (chạy lệnh dạng list, không shell). **Repo Agent không có test nào**, nên "test gốc" không có để chép; thay bằng test mới kiểm tra đúng hành vi của `agent_runtime.py` và `gateway.py` gốc cùng các chỗ sửa an toàn. Bằng chứng: `tests/test_m8_runtime.py` (14 test, gồm chuỗi chèn lệnh `;` `&&` `|` `$()` backtick, xuống dòng chỉ được in ra như chữ).
- [x] 4. Test cũ và test của runtime đều xanh: 133 test chạy qua; compileall và secret-scan sạch.

## M9: Gộp repo Agent, phần 2: tool chạy lệnh
- [x] 1. `TerminalTool` (`local_ai/runtime/terminal.py`) dựng trên `run_command` của M8:
  - chỉ chạy lệnh có trong allowlist ở `configs/tools/terminal.json` (đúng danh sách `ALLOWED_PREFIXES` của Agent gốc);
  - không dùng shell, tham số truyền dạng list, có timeout;
  - ghi log JSONL từng lệnh, kể cả lệnh bị từ chối;
  - chặn thêm tham số nguy hiểm và đường dẫn ra ngoài thư mục làm việc.
  - Bổ sung khi rà lại: chặn `date -s`/`--set` (đổi đồng hồ hệ thống khi chạy bằng root) và tham số ngắn viết gộp (`date -us...`, `tail -qf`).

  Bằng chứng: `AllowedCommandTests`, `InjectionTests.test_dangerous_options_and_paths_are_rejected`; test M8 kiểm tra bằng AST rằng không nơi nào trong `local_ai/` gọi với `shell=True`.
- [x] 2. Tool tắt mặc định (`"enabled": false`). Muốn bật phải khai báo rõ: `enabled: true` trong cấu hình, hoặc `TerminalTool(enabled=True)`. Khi đang tắt, mọi lần gọi bị từ chối kèm lý do. Bằng chứng: `ConfigTests`, `AgentRegistrationTests.test_disabled_terminal_fails_safely`.
- [x] 3. Gateway HTTP (`local_ai/runtime/gateway.py`, `configs/runtime/gateway.json`):
  - tắt mặc định;
  - chỉ nghe `127.0.0.1` (`0.0.0.0` hay `localhost` đều bị từ chối);
  - bắt buộc token dài ít nhất 16 ký tự, đọc từ biến môi trường; thiếu token thì không bật;
  - không có endpoint chạy lệnh qua mạng.

  Bằng chứng: `GatewayTests` (2 test).
- [x] 4. Đăng ký vào `ToolRegistry` bằng `register_terminal`; agent gọi `{"tool": "terminal", ...}` và hoàn thành việc; tool đang tắt thì agent nhận lỗi, không sập. Bằng chứng: `AgentRegistrationTests` (2 test).
- [x] 5. Có test chặn chèn lệnh: `;` `&&` `&` `|` `$()` backtick, xuống dòng, `\r`, `<` `>`, cả dạng chuỗi lẫn dạng list, và không lệnh nào được chạy. Lệnh ngoài allowlist (`rm`, `curl`, `bash`, `/bin/ls`, `git push`...) bị từ chối. Timeout dừng lệnh `sleep 30` sau 0,5 giây. Bằng chứng: `InjectionTests` (3 test), `AllowedCommandTests.test_timeout_stops_the_command`.

## M10: Notebook train trên Colab free (0.5B)
- [x] 1. `notebooks/train_colab.ipynb` (sinh từ `notebooks/build.py`). README có nút "Open in Colab" ở đầu file và ở mục "Train trên Colab miễn phí", trỏ tới notebook ở nhánh `main`. Bằng chứng: `ReadmeTests`.
- [x] 2. 9 ô code đúng thứ tự:
  1. kiểm tra GPU (không có thì hướng dẫn chọn T4);
  2. clone repo, cài thư viện, không cài lại torch;
  3. đọc `HF_TOKEN` từ Colab Secrets (`userdata.get`);
  4. `hf-sft --total 2000` từ 3 preset;
  5. chấm model gốc `smoke`;
  6. QLoRA 4bit fp16 (`configs/training/colab_smoke.json`);
  7. chấm lại `smoke-colab` (mục mới trong danh sách model, trỏ tới adapter vừa train);
  8. in bảng so sánh bằng lệnh mới `python -m local_ai.evaluation.compare`;
  9. đẩy adapter lên repo riêng tư (`python -m local_ai.training.hub push-adapter`).

  Bằng chứng: `NotebookTests`, `CompareTests`, `HubTests.test_push_adapter_creates_private_repo_and_uploads`.
- [x] 3. Finetune thêm `--push-to-hub --hub-model-id` (khóa `push_to_hub`, `hub_model_id`, `hub_private`): truyền `hub_strategy="checkpoint"` cho `SFTConfig`. Chạy lại mà máy không còn `checkpoint-N` thì tải `last-checkpoint` về `output_dir/_hub/` rồi train tiếp. Repo chưa có checkpoint thì train từ đầu; lỗi khác (mất mạng, sai quyền) thì báo ra, không lặng lẽ train lại. Thêm `dtype` (ghi đè dtype, `float16` cho T4). Bằng chứng: `FinetuneHubTests`, `HubTests`.
- [x] 4. Notebook hợp lệ theo `nbformat.validate` (nbformat 5.11.1), không có output, `execution_count` rỗng. `pip install` ghim `==` cho mọi gói, không có torch. Mỗi ô code mở đầu bằng chú thích `# Bước N: ...` tiếng Việt. Không có token trong notebook. Bằng chứng: `NotebookTests`.
- [x] 5. `test_notebook_commands_run_with_dry_run` lấy mọi dòng `!python -m local_ai...` trong notebook, thêm `--dry-run` rồi chạy thật (với `HF_HUB_OFFLINE=1`). Thêm `--dry-run` cho `hf-sft`, eval và `compare`. Test kiểm tra thêm:
  - dữ liệu đủ 2000 dòng;
  - QLoRA fp16 đẩy vào đúng repo;
  - so sánh đúng 2 báo cáo;
  - adapter được đẩy đúng là adapter vừa train và vừa chấm.

  160 test chạy qua; compileall và secret-scan sạch.

Chưa kiểm chứng: notebook **chưa chạy trên Colab thật** (môi trường phát triển không có GPU); thời gian 20–40 phút ghi trong notebook là ước đoán.

## M11: Colab cho Qwen3-4B + hướng dẫn iPhone
- [x] 1. `configs/training/colab_light.json`: model `light` (Qwen3-4B), QLoRA 4bit, fp16, batch 1 (tích lũy 16 bước), `save_steps` 10.
  - `max_length` là 2048: VRAM ước tính 3,8 GB trên 15 GB của T4, và 2048 đúng bằng độ dài mà `vram.py` giả định.
  - Đo trên 2000 dòng thật của 3 preset: cắt ở 2048 thì 16 dòng bị cắt; cắt ở 1024 thì tới 308 dòng.
  - Thêm mục `light-colab` (adapter vừa train, `max_new_tokens` 512 để Qwen3 có chỗ cho phần `<think>`).

  Bằng chứng: `LightConfigTests`.
- [x] 2. Notebook có ô **Bước 1: chọn model**: form Colab `MODEL = "smoke"  # @param ["smoke", "light"]`. Mọi lệnh sau đó dùng `colab_{MODEL}.json`, `{MODEL}-colab` và repo `huyen-{MODEL}-qlora` riêng cho từng model.
  - **Ước tính thời gian:** ô **Bước 6** chạy lệnh mới `python -m local_ai.training.estimate` (`local_ai/training/estimate.py`). Lệnh in số bước, số phút train, số phút chấm, VRAM so với T4, và số bước đã train nếu có checkpoint trên máy hoặc trên Hub.
  - Ước tính theo 2000 dòng: `smoke` khoảng 21 phút, `light` khoảng 109 phút, chưa tính cài đặt. Đây là ước lượng thô, chưa đo trên T4 thật.
  - **Tự train tiếp khi Colab ngắt:** lệnh train đẩy checkpoint lên Hub, chạy lại thì tải `last-checkpoint` về (từ M10); ô ước tính báo "Đã train N/125 bước".

  Bằng chứng: `NotebookChoiceTests`, `EstimateTests`.
- [x] 3. `docs/TRAIN_COLAB.md` hướng dẫn trên iPhone:
  - chuẩn bị token Write, tắt tự khóa màn hình;
  - mở link (bật "Yêu cầu trang web cho máy tính" trong Safari);
  - chọn T4, thêm `HF_TOKEN` vào Secrets (bật Notebook access);
  - chọn model, Run all (các hộp thoại sẽ gặp);
  - kết quả nằm đâu, khi Colab ngắt thì làm gì;
  - bảng lỗi hay gặp.

  README và notebook có link tới tài liệu này. Bằng chứng: `GuideTests` (kiểm tra đủ các bước, và số phút trong tài liệu và README khớp với ước tính).
- [x] 4. `test_light_notebook_commands_run_with_dry_run` chạy mọi lệnh của notebook với `MODEL = "light"` bằng `--dry-run`. Test M10 được sửa theo cấu trúc notebook mới (thêm 2 ô, đánh số lại, thay biến `{MODEL}`, `{MAX_NEW_TOKENS}`); không bỏ kiểm tra nào. 171 test chạy qua; compileall và secret-scan sạch.

Chưa kiểm chứng: notebook **chưa chạy trên Colab thật**; thời gian và VRAM là ước lượng.

## M12: Chất lượng dữ liệu
- [x] 1. `local_ai/data/quality.py`, chỉ dùng thư viện chuẩn; ngưỡng ở `configs/datasets/quality.json`. `hf-sft` bật mặc định (tắt bằng `--no-quality`), `build` bật bằng `--quality`. Các bộ lọc chạy sau bước loại trùng tuyệt đối, trước bước chặn trùng eval:
  - **ngôn ngữ** (ưu tiên tiếng Việt):
    - nhận tiếng Việt trước tiên, theo chữ có dấu, nên câu trộn với code hay tiếng Anh vẫn tính là tiếng Việt;
    - loại chữ không phải Latin, tiếng Việt không dấu, và dòng khác ngôn ngữ nguồn khai báo (preset `vietnamese` phải là tiếng Việt có dấu);
    - dòng tiếng Việt trong nguồn khác vẫn được giữ;
  - **độ dài:** câu hỏi dưới 3 ký tự, câu trả lời dưới 2 ký tự, hội thoại quá 16.000 ký tự;
  - **lặp từ/câu:** xét từng lượt riêng, bỏ qua khối code và lệnh LaTeX. Loại khi có từ lặp liền quá 8 lần, hoặc khi hơn 50% số câu hay cụm 5 từ là bản lặp;
  - **gần trùng:** MinHash tự viết (kiểu một hoán vị, 128 ngăn), LSH 32 dải, so lại bằng Jaccard thật trên cụm 3 từ, ngưỡng 0,8.

  Thử trên 2000 dòng thật của 3 preset: chạy khoảng 5 giây, chỉ loại 1 dòng (lời giải toán lặp công thức 60%). Bản đầu dùng MinHash 128 hoán vị mất 21 giây và bắt nhầm lệnh LaTeX, bảng số, hội thoại nhiều lượt; đã sửa như trên.
- [x] 2. `manifest.json` có mục `quality`:
  - `before` / `after`: số dòng, ngôn ngữ phát hiện được, tỉ lệ tiếng Việt, độ dài;
  - `removed`: số dòng mỗi bộ lọc đã loại;
  - `config`: cấu hình đã dùng.

  Khi tắt lọc thì ghi `{"enabled": false}`. Dòng bị loại nằm trong `rejected.jsonl` kèm `quality_filter` và `quality_reason`. Bằng chứng: `PipelineTests`.
- [x] 3. Fixture `tests/fixtures/quality/{language,length,repetition,near_duplicate}.jsonl`; mỗi dòng ghi kết quả mong đợi. Có cả dòng đối chứng phải giữ: code lặp, LaTeX lặp, dãy số 0, hội thoại nhắc lại qua nhiều lượt, tiếng Việt trong preset code. Đã kiểm tra: nếu bỏ phần xử lý tương ứng thì các dòng đối chứng này bị loại nhầm.

  Bằng chứng: `LanguageFilterTests`, `LengthFilterTests`, `RepetitionFilterTests`, `NearDuplicateTests` (kể cả MinHash ước lượng Jaccard lệch dưới 0,2), `ConfigTests`. 189 test chạy qua; compileall và secret-scan sạch.

## M13: Agent chạy model thật trên Colab
- [x] 1. `notebooks/agent_colab.ipynb` (sinh từ `notebooks/build.py`):
  - cài Ollama bản ghim `OLLAMA_VERSION=0.34.4`, bản mới nhất lúc viết, cùng `zstd` mà bản cài `.tar.zst` cần;
  - chạy `ollama serve` ở nền, đợi tới khi trả lời;
  - kéo `qwen3:4b` (2,5 GB, đã kiểm tra tag còn trên kho Ollama);
  - agent gọi model qua adapter OpenAI của M3, dùng mục mới `ollama-colab` (`http://localhost:11434/v1`).

  Phần agent chỉ dùng thư viện chuẩn, không `pip install`. Bằng chứng: `OpenAIAdapterTests.test_five_tasks_through_openai_compatible_server` (5 nhiệm vụ qua server HTTP giả nói chuẩn OpenAI).
- [x] 2. Lệnh mới `python -m local_ai.agents.tasks` chạy 5 nhiệm vụ trong `data/eval/agent_tasks_v1.jsonl`:
  - 2 nhiệm vụ calculator, `ls`, `cat`, và `cat` rồi calculator;
  - mỗi nhiệm vụ chạy trong thư mục làm việc riêng, chép từ `data/eval/agent_workspace/`, và chỉ bật TerminalTool trong đó;
  - in trace và tỉ lệ thành công;
  - nhiệm vụ đạt khi câu trả lời đúng **và** agent đã gọi mọi công cụ cần dùng;
  - `--scripted` chạy bằng câu trả lời mẫu, ra 5/5.

  Sửa thêm để model thật dùng được công cụ:
  - `ToolRegistry` có mô tả công cụ;
  - agent gửi danh sách công cụ, lịch sử quan sát và mẫu JSON trong prompt;
  - agent hiểu `arguments` dạng chuỗi JSON.

  Bằng chứng: `TaskTests` (có ca sai đáp án, thiếu công cụ, JSON hỏng, thử `cat /etc/passwd` và `rm`), `ToolDescriptionTests`.
- [x] 3. Notebook hợp lệ theo nbformat, khớp `build.py`, không có output, mỗi ô code có chú thích `# Bước N:`, không có token. `test_notebook_commands_run_with_dry_run` chạy 2 lệnh `!python -m local_ai...` của notebook bằng `--dry-run`. 200 test chạy qua; compileall và secret-scan sạch.

Chưa kiểm chứng: **chưa chạy với Ollama và `qwen3:4b` thật** (không tải model trong môi trường phát triển), nên chưa biết tỉ lệ thành công thật; thời gian 5–15 phút là ước đoán.

## M14: Tổng kết tuần 2
- [x] 1. Cập nhật README, `docs/ARCHITECTURE.md` và lộ trình.
  - README:
    - bảng "Trạng thái sau tuần 1 và tuần 2" (thêm 2 dòng Runtime và Notebook Colab, ghi rõ phần nào chưa chạy thật);
    - 2 nút Open in Colab ở đầu file;
    - lộ trình đề xuất 7 mốc tuần 3 (M15–M21), gom các việc còn lại của tuần 1 và 2.
  - ARCHITECTURE: phạm vi, luồng chạy trên Colab và luồng agent + Ollama, phần kiểm thử theo mốc, danh sách phần chưa kiểm chứng.
- [x] 2. `memory.md`:
  - % tiến độ từng phần (tính theo hạng mục đã chạy thật);
  - việc chủ repo tự làm (chạy 2 notebook, token HF quyền Write, archive repo Agent cũ, PR #4...);
  - 7 mốc đề xuất cho tuần 3, giống README.
- [x] 3. README khớp code: `tests/test_m14_summary.py`, cùng test cờ lệnh của M7.
  - Mọi module chạy được bằng `python -m` đều có trong README (bổ sung `local_ai.training.hub`).
  - Mọi notebook có nút Colab trỏ đúng file.
  - README nhắc tới mọi cấu hình train, `quality.json`, `terminal.json`, `gateway.json`, file nhiệm vụ agent và mọi tài liệu trong `docs/`.
  - Bảng tiến độ tuần 2 đủ 7 mốc.
- [x] 4. Mọi test xanh: 206 test chạy qua; compileall và secret-scan sạch.

---

# Nhiệm vụ tuần 1 (M1–M7, đã xong)

**Đã xong cả 7 mốc và đã gộp vào `main` ngày 24/9 (PR #6, commit gộp `63bdc61`).**

Bắt đầu: 2026-09-24, từ `main` tại commit `60e8541` (28 test chạy qua, 1 bỏ qua; compileall và secret-scan sạch).

Cách làm: mỗi lượt làm **tối đa 1 mốc**, theo thứ tự M1 → M7, bằng skill `lam-moc` (gõ `/lam-moc` trong Claude Code hoặc nói "làm mốc tiếp theo"). Mốc đang làm, việc dở và lỗi gặp được ghi trong `memory.md`.

## Cách chấm
- Mỗi mốc có danh sách tiêu chí `[ ]`. Chỉ đánh `[x]` khi tiêu chí đã đạt và có bằng chứng (tên test hoặc lệnh kiểm tra).
- % của mốc = số tiêu chí đạt ÷ tổng số tiêu chí của mốc. Tổng % = trung bình cộng 7 mốc.
- Mốc chỉ được ghi **Xong** khi đạt đủ 100% tiêu chí **và** cả ba lệnh kiểm tra đều xanh:
  `python -m unittest discover -s tests -v`, `python -m compileall -q local_ai`, `python -m local_ai.data secret-scan`.
- Test không được tải model hay dataset từ mạng.

## Bảng tiến độ

| Mốc | Nội dung | Ngày gợi ý | Trạng thái | Tiến độ | Bằng chứng (test, commit) |
| --- | --- | --- | --- | --- | --- |
| M1 | Agent không sập khi công cụ lỗi; dọn repo | Ngày 1 | **Xong** | 9/9 (100%) | `tests/test_m1_agent_cleanup.py` (11 test), commit `ebf3c63` |
| M2 | Model ảnh+chữ, nén 4-bit (QLoRA), ước tính VRAM | Ngày 2 | **Xong** | 8/8 (100%) | `tests/test_m2_models.py` (20 test), commit `d8cc5ae` |
| M3 | Adapter gọi server local kiểu OpenAI | Ngày 3 | **Xong** | 7/7 (100%) | `tests/test_m3_local_server.py` (17 test), commit `fa88440` |
| M4 | Preset dataset Hugging Face | Ngày 4 | **Xong** | 9/9 (100%) | `tests/test_m4_presets.py` (15 test) + chạy thật, commit `ed816ca` |
| M5 | Đánh giá (eval) mở rộng | Ngày 5 | **Xong** | 8/8 (100%) | `tests/test_m5_eval.py` (20 test), commit `854faa1` |
| M6 | Test chạy thật trên CPU với model tí hon | Ngày 6 | **Xong** | 7/7 (100%) | `tests/test_m6_cpu_pipeline.py` (3 test, chạy thật khoảng 6 giây), commit `2be8dc6` |
| M7 | Tổng kết | Ngày 7 | **Xong** | 7/7 (100%) | `tests/test_m7_summary.py` (5 test), commit `f9116f0` |
| **Tổng** | | | | **100%** (7/7 mốc) | 114 test chạy qua, không bỏ qua test nào; compileall, secret-scan sạch |

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
  - Cập nhật 26/9 (M15, chủ repo đồng ý): số đo thật của Qwen3-4B cho thấy công thức cũ ước tính thấp khoảng 2 lần. Công thức mới ước tính train QLoRA 27B khoảng 34,6 GB (trên 24 GB), test sửa theo.

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

**Tiêu chí xong** (thêm tiêu chí 6–7 ngày 24/9 theo yêu cầu của chủ repo: cập nhật `docs/ARCHITECTURE.md`; `memory.md` có % từng phần và lệnh train khi có GPU)
- [x] 1. README đúng trạng thái thật: có bảng trạng thái (phần nào chạy thật, phần nào mới test bằng module giả), cài đặt, luồng làm việc, lệnh từng bước.
  - Mọi lệnh của repo trong README đã chạy thử trong lượt này: `vram`, `demo` (có và không có `--model`), `list-presets`, `hf-sft --config presets/code.json` (thật, 1000 dòng), `hf-sft --preset ...` (thật, 2500 dòng), `finetune --dry-run` / chạy thật (trả `skipped` vì không có GPU), `evaluation --scripted` / `--model ollama` / `--train-data`, test M6.
  - Không chạy được: lệnh của phần mềm ngoài (`ollama`, `llama-server`, `vllm`), lệnh train trên GPU, và lệnh eval phải tải trọng số model. Lý do: không có GPU, và quy tắc không cho tải trọng số model. README ghi rõ các lệnh này "chưa chạy thử".
  - Test tự kiểm tra README: 25 lệnh `python -m local_ai...` chỉ dùng cờ có thật; số VRAM khớp `vram.py`; `adapter_path` khớp thư mục đầu ra của train.
  - Bằng chứng: `tests/test_m7_summary.py` `ReadmeMatchesCodeTests` (4 test).
- [x] 2. Có lộ trình sau tuần 1, ghi rõ là đề xuất, chưa làm: mục "Lộ trình tiếp theo (đề xuất, chưa làm)" trong README, 6 việc. Bằng chứng: `ReadmeMatchesCodeTests.test_readme_has_status_gpu_plan_and_proposed_roadmap`.
- [x] 3. Chấm lại % từng mốc trong bảng tiến độ theo tiêu chí, kèm bằng chứng (tên test, commit), và ghi tổng %: 100%.
- [x] 4. `memory.md` có tổng kết tuần: việc đã xong, việc còn dở, lỗi còn tồn.
- [x] 5. Cả ba lệnh kiểm tra (test, compileall, secret-scan) đều xanh: 114 test chạy qua.
- [x] 6. `docs/ARCHITECTURE.md` viết lại theo code hiện tại: luồng dữ liệu → train → eval, bảng file cấu hình, từng module và lệnh, giới hạn chưa kiểm chứng.
- [x] 7. `memory.md` có % tiến độ từng phần, lệnh train khi có GPU (smoke → light → primary) và VRAM cần cho từng bước; README có bảng tương ứng. `platform.json` có sẵn `smoke-lora`, `light-lora`, `primary-qlora` trỏ tới adapter của từng bước để eval. Bằng chứng: `ReadmeMatchesCodeTests.test_post_training_models_point_to_training_outputs`, `test_vram_numbers_in_readme_match_estimator`.

**Sửa thêm khi chạy thử lệnh trong README:**
- Preset `reasoning` có dòng mà `uuid` là chuỗi giữ chỗ `"NaN"`. Các dòng đó bị coi là trùng id và bị loại oan (2/750). Nay id rỗng, NaN, hoặc chuỗi `"NaN"`/`"None"`/`"null"` được thay bằng số thứ tự dòng; chạy lại giữ đủ 2500/2500 dòng. Bằng chứng: `tests/test_m4_presets.py` `PresetMappingTests.test_missing_or_nan_id_falls_back_to_row_number`.
- `secret-scan` quét cả dữ liệu tải về đã nằm trong `.gitignore`, nên báo nhầm vì code mẫu trong dataset có dòng dạng `password = ...`. Nay chỉ quét file có thể bị commit (theo `git ls-files`); thư mục không phải repo git thì vẫn quét hết như cũ. Bằng chứng: `SecretScanTests.test_gitignored_files_are_not_scanned`.
