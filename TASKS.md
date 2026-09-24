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
| M1 | Agent không sập khi công cụ lỗi; dọn repo | Ngày 1 | Chưa làm | 0/9 (0%) | — |
| M2 | Model ảnh+chữ, nén 4-bit (QLoRA), ước tính VRAM | Ngày 2 | Chưa làm | 0/7 (0%) | — |
| M3 | Adapter gọi server local kiểu OpenAI | Ngày 3 | Chưa làm | 0/7 (0%) | — |
| M4 | Preset dataset Hugging Face | Ngày 4 | Chưa làm | 0/6 (0%) | — |
| M5 | Đánh giá (eval) mở rộng | Ngày 5 | Chưa làm | 0/7 (0%) | — |
| M6 | Test chạy thật trên CPU với model tí hon | Ngày 6 | Chưa làm | 0/6 (0%) | — |
| M7 | Tổng kết | Ngày 7 | Chưa làm | 0/5 (0%) | — |
| **Tổng** | | | | **0%** | |

## Ngoài phạm vi tuần này
Không làm: pretrain/corpus quy mô lớn, huấn luyện phân tán, xuất GGUF, gọi API trả phí, tải trọng số model hay dataset lớn. Việc phát sinh ngoài 7 mốc: ghi vào "Việc dở" trong `memory.md` và hỏi chủ repo trước.

---

## M1: Agent không sập khi công cụ lỗi; dọn repo

**Hiện trạng** (đã chạy thử trên `main` ngày 24/9):
- `calculator("1/0")` ném `ZeroDivisionError` và `calculator("2 +")` ném `SyntaxError` ra ngoài, làm sập cả agent, vì `ToolRegistry.execute` chỉ bắt `TypeError` và `ValueError`.
- `_parse_json_object` trả về `{}` khi JSON có chữ thừa, nằm trong khối ```` ```json ````, hoặc đứng sau `<think>`.

**Tiêu chí xong**

Agent:
- [ ] 1. Công cụ ném bất kỳ ngoại lệ nào (kể cả `ZeroDivisionError`, `SyntaxError`) thì `ToolRegistry.execute` trả kết quả thất bại kèm thông báo lỗi, không ném ra ngoài. Có test với `1/0` và `2 +`.
- [ ] 2. Agent chạy hết vòng, không sập khi công cụ lỗi: lỗi được ghi vào trace và model được thử lại. Có test: model giả gọi `1/0`, sau đó sửa thành phép tính đúng, và agent hoàn thành.
- [ ] 3. Tách được JSON từ câu trả lời model trong các trường hợp: JSON thuần; có chữ thừa trước hoặc sau; nằm trong khối ```` ```json ```` hoặc ```` ``` ````; đứng sau `<think>...</think>` (bỏ qua JSON nằm trong `<think>`); chuỗi trong JSON chứa dấu `{` `}`. Mỗi trường hợp có một test.
- [ ] 4. Agent hoàn thành nhiệm vụ khi model giả trả lời theo các kiểu ở tiêu chí 3.

Dọn repo:
- [ ] 5. Cất phần corpus 10T vào `archive/corpus-10t/`: `local_ai/data/corpus.py`, `configs/datasets/corpus_10t.json`, `configs/datasets/smoke_real.json`, các lệnh CLI corpus và test đi kèm. Code trong `local_ai/` không còn import phần này, và test đã cất không còn chạy trong `tests/`.
- [ ] 6. Lệnh `python -m local_ai.data secret-scan` vẫn chạy: hàm quét được tách ra một file riêng trong `local_ai/` và có test.
- [ ] 7. Không còn khóa cấu hình thừa: mọi khóa trong `configs/**/*.json` (trừ `_comment`) đều được code đọc, có test tự dò. Khóa thừa hiện chỉ nằm trong `corpus_10t.json` và `smoke_real.json`.
- [ ] 8. Không còn link `huyenb2404-ops` (hiện nằm ở `data/raw/seed_examples.jsonl` và `data/eval/seed_eval.jsonl`); đổi thành `https://github.com/hytmk2912/Huyen`.
- [ ] 9. README khớp hướng fine-tune: mô tả luồng dataset Hugging Face → sft.jsonl → LoRA/QLoRA → đánh giá, bỏ mục tiêu 10T token, phần lộ trình trỏ tới `TASKS.md`.

---

## M2: Model ảnh+chữ, nén 4-bit (QLoRA), ước tính VRAM

**Hiện trạng:**
- Hugging Face gắn nhãn `image-text-to-text` (kiến trúc `qwen3_5`) cho model chính `huihui-ai/Huihui-Qwen3.8-27B-abliterated`, nhưng adapter chỉ nạp bằng `AutoModelForCausalLM`. Chưa chạy thật để xác nhận model có nạp được không.
- Trường `quantization` trong `ModelConfig` có khai báo nhưng không code nào đọc. Chưa có QLoRA.

**Tiêu chí xong**
- [ ] 1. Cấu hình model khai báo được loại đầu vào (ví dụ `modality: "text"` hoặc `"image-text"`), và model chính được khai báo đúng loại theo thẻ trên Hugging Face.
- [ ] 2. Adapter chọn đúng lớp nạp: `AutoModelForImageTextToText` + `AutoProcessor` cho model ảnh+chữ, `AutoModelForCausalLM` + `AutoTokenizer` cho model chữ. Có test dùng module `transformers` giả, không tải model.
- [ ] 3. Gửi được tin nhắn gồm ảnh (đường dẫn file cục bộ) và chữ tới model ảnh+chữ. Gửi ảnh tới model chỉ có chữ thì báo lỗi rõ ràng bằng tiếng Việt. Có test.
- [ ] 4. Khai báo `quantization: "nf4"` trong cấu hình model thì khi chạy suy luận, model được nạp 4-bit bằng bitsandbytes. Có test dùng module giả.
- [ ] 5. `finetune` có `method: "qlora"`: nạp model 4-bit, chuẩn bị cho k-bit training rồi gắn LoRA. Thiếu GPU hoặc bitsandbytes thì trả về `"skipped"`. Có file cấu hình mẫu cho model chính và có test.
- [ ] 6. Có lệnh ước tính VRAM (ví dụ `python -m local_ai.models.vram`) in bảng cho **mọi** model trong `configs/models/platform.json`. Bảng gồm số tham số, VRAM cho trọng số ở bf16 và 4-bit, và ước tính khi train LoRA/QLoRA. Lệnh chạy không cần mạng và ghi rõ đây là ước lượng.
- [ ] 7. Có test kiểm tra số liệu ước tính, ví dụ model 27B: bf16 ≈ 54 GB, 4-bit ≈ 14–16 GB.

Tham khảo: PR #4 (commit `59230b9`, chưa gộp) đã có code QLoRA và ước lượng bộ nhớ, có thể dùng lại.

---

## M3: Adapter gọi server local kiểu OpenAI (Ollama, llama.cpp)

**Tiêu chí xong**
- [ ] 1. Adapter gửi `POST {base_url}/chat/completions` theo chuẩn OpenAI và trả về `choices[0].message.content`. Dùng được với Ollama (`http://localhost:11434/v1`) và llama.cpp server (`http://localhost:8080/v1`). Chỉ dùng thư viện chuẩn của Python, không thêm thư viện `openai`.
- [ ] 2. Khai báo được trong `configs/models/platform.json` (ví dụ `backend: "openai_compatible"`, `base_url`, tên model trên server). Khóa truy cập, nếu server yêu cầu, chỉ đọc từ biến môi trường.
- [ ] 3. Mặc định chỉ cho phép địa chỉ local hoặc mạng nội bộ; địa chỉ bên ngoài bị từ chối, để không vô tình gọi API trả phí.
- [ ] 4. Lỗi mạng, hết thời gian chờ hay mã lỗi HTTP đều được báo rõ ràng bằng tiếng Việt; agent không sập khi server lỗi.
- [ ] 5. Router và agent dùng được adapter này qua cấu hình.
- [ ] 6. Có test dựng server giả bằng `http.server` trên localhost cho ba trường hợp: trả lời đúng, trả mã 500, và chậm quá thời gian chờ. Test không cần Ollama hay llama.cpp thật.
- [ ] 7. README có hướng dẫn chạy với Ollama và llama.cpp: lệnh khởi động server và cấu hình mẫu.

---

## M4: Preset dataset Hugging Face (code, reasoning, tiếng Việt)

**Tiêu chí xong**
- [ ] 1. Có 2–3 file preset trong `configs/datasets/presets/` (code, reasoning, tiếng Việt). Mỗi file trỏ tới một dataset có thật trên Hugging Face (`org/name`, có `revision`).
- [ ] 2. Mỗi preset ghi giấy phép theo trang dataset, kèm trạng thái `"cần kiểm tra lại"`. README nhắc kiểm tra lại giấy phép trước khi dùng.
- [ ] 3. Mỗi preset bật `streaming: true` và có `limit` nhỏ (mặc định ≤ 1000 dòng): chỉ đọc đúng số dòng đó, không tải cả dataset.
- [ ] 4. Chạy được bằng `python -m local_ai.data hf-sft --config configs/datasets/presets/<tên>.json`.
- [ ] 5. File `sft.jsonl` được xuất ra đúng: hội thoại nhiều lượt giữ nguyên (kể cả system), reasoning và context không bị mất. Lỗi này hiện có trên `main`, xem mục "Lỗi gặp" trong `memory.md`.
- [ ] 6. Có test dùng loader giả (không mạng) cho từng preset: ánh xạ cột đúng, bản ghi hợp lệ, loader nhận `streaming=True` và chỉ đọc `limit` dòng.

Tham khảo: PR #4 (commit `2626c7d`, chưa gộp) đã sửa lỗi hội thoại nhiều lượt và lỗi mất reasoning/context.

---

## M5: Đánh giá (eval) mở rộng

**Hiện trạng:** `local_ai/evaluation/benchmarks.py` mới chỉ chấm kiểu khớp đúng từng chữ.

**Tiêu chí xong**
- [ ] 1. Chấm kiểu "chứa đáp án": không phân biệt chữ hoa/thường, có chuẩn hóa khoảng trắng và Unicode tiếng Việt.
- [ ] 2. Chấm bằng regex.
- [ ] 3. Chấm bài code bằng unit test: chạy code model viết cùng với test trong sandbox có giới hạn thời gian. Code sai cú pháp, lỗi khi chạy hoặc lặp vô hạn thì tính là không đạt, chương trình không sập.
- [ ] 4. Có bộ khoảng 30 câu trong `data/eval/`, gồm cả tiếng Việt và tiếng Anh (mỗi thứ tiếng ít nhất 10 câu) và ít nhất 5 bài code có unit test. Mỗi câu ghi rõ ngôn ngữ, nhóm và cách chấm.
- [ ] 5. Báo cáo kết quả theo nhóm và theo ngôn ngữ, liệt kê các câu sai.
- [ ] 6. Có test cho từng cách chấm, cả trường hợp đạt và không đạt, kể cả code lặp vô hạn.
- [ ] 7. Không có câu eval nào lọt vào dữ liệu train: kiểm tra trùng theo nội dung câu hỏi, không chỉ theo ID. Lỗi chỉ so ID hiện có trên `main`.

Tham khảo: PR #4 (chưa gộp) có `local_ai/evaluation/suites.py` (chấm khớp đúng, chứa đáp án, so số) và phần chặn rò rỉ eval theo nội dung (commit `2626c7d`).

---

## M6: Test chạy thật trên CPU với model tí hon

**Tiêu chí xong**
- [ ] 1. Test tự tạo tokenizer và model tí hon khởi tạo ngẫu nhiên (ví dụ 2 lớp, kích thước ẩn 32), không tải model, tokenizer hay dataset nào. Test chạy với `HF_HUB_OFFLINE=1` để chắc chắn điều đó.
- [ ] 2. Phần dữ liệu: từ file mẫu trong repo, qua bước build, ra `sft.jsonl`.
- [ ] 3. Train LoRA đúng 2 bước trên CPU qua `local_ai.training.finetune` (có cấu hình cho phép chạy không cần GPU) và tạo ra adapter.
- [ ] 4. Phần eval: chạy bộ đánh giá với model vừa train và ghi `eval_report.json`. Điểm thấp là bình thường.
- [ ] 5. Toàn bộ test chạy dưới 3 phút trên CPU; tự bỏ qua (skip, không báo lỗi) khi máy thiếu torch, transformers, peft hoặc trl.
- [ ] 6. Đã chạy thật ít nhất một lần trên máy có đủ thư viện, và ghi thời gian chạy cùng loss vào `memory.md`.

---

## M7: Tổng kết

**Tiêu chí xong**
- [ ] 1. README cập nhật đúng trạng thái thật: cài đặt, luồng làm việc và lệnh cho từng bước. Mọi lệnh trong README đều đã chạy thử được.
- [ ] 2. Có lộ trình sau tuần 1: vài việc tiếp theo, ghi rõ là đề xuất, chưa làm.
- [ ] 3. Chấm lại % từng mốc trong bảng tiến độ theo tiêu chí, kèm bằng chứng (tên test, commit), và ghi tổng %.
- [ ] 4. `memory.md` có tổng kết tuần: việc đã xong, việc còn dở, lỗi còn tồn.
- [ ] 5. Cả ba lệnh kiểm tra (test, compileall, secret-scan) đều xanh.
