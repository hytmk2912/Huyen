# Giấy phép các dataset preset

Ngày đọc: 25/9/2026 (3 preset đầu) và 30/9/2026 (`vietnamese_aya`, M26), trên dataset card (trang giới thiệu dataset) ở Hugging Face. Mỗi preset khóa đúng một commit (`revision` trong `configs/datasets/presets/*.json`), nên nội dung dưới đây ứng với commit đó.

**Lưu ý:** đây là kết quả đọc tài liệu công khai, không phải tư vấn pháp lý. Trạng thái trong preset vẫn để "cần kiểm tra lại trên dataset card": chỉ chủ repo mới quyết định được có chấp nhận các điều kiện dưới đây hay không.

## Tóm tắt
| Preset | Dataset | Giấy phép ghi trên card | Nguồn gốc dữ liệu | Dùng được cho |
| --- | --- | --- | --- | --- |
| `code` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | ODC-By | Hàm Python mẫu lấy từ quy trình MultiPL-T; đề bài và lời giải do StarCoder2-15B tự sinh, lời giải đã chạy thử | Cá nhân và thương mại, **phải ghi nguồn** |
| `reasoning` | `open-r1/OpenR1-Math-220k` | Apache-2.0 | Đề bài từ NuminaMath 1.5; lời giải do DeepSeek R1 sinh; kiểm tra bằng Math Verify, 12% dùng Llama-3.3-70B-Instruct làm giám khảo | Cá nhân và thương mại; nên ghi nguồn |
| `vietnamese` | `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` | Apache-2.0 | Bản dịch tiếng Việt của ChatAlpaca; ChatAlpaca sinh bằng ChatGPT (gpt-3.5-turbo) từ dữ liệu Stanford Alpaca | **Chỉ dùng cá nhân hoặc nghiên cứu, không dùng thương mại** (xem dưới) |
| `vietnamese_aya` | `CohereLabs/aya_dataset` | Apache-2.0 | Người tình nguyện của Aya Open Science Initiative (Cohere Labs) tự viết câu hỏi và câu trả lời; preset chỉ lấy dòng tiếng Việt loại `original-annotations` | Cá nhân và thương mại; nên ghi nguồn |

## `code`: phải ghi nguồn
- ODC-By (Open Data Commons Attribution) cho phép dùng, sửa, chia sẻ, kể cả thương mại, với điều kiện ghi nguồn dataset.
- Nếu công khai model hoặc dữ liệu đã train, hãy ghi: "Dữ liệu code: bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-By)".

## `reasoning`: Apache-2.0
- Card ghi rõ "The dataset is licensed under Apache 2.0".
- Lời giải do DeepSeek R1 sinh (DeepSeek R1 dùng giấy phép MIT). Một phần được chấm bằng Llama-3.3-70B; phần này chỉ là giám khảo, không phải lời giải.

## `vietnamese`: không dùng thương mại
- Card chỉ có một dòng giấy phép (Apache-2.0), không mô tả nguồn gốc.
- Tên file dữ liệu là `vi_chatalpaca_cleaned.json`: đây là bản dịch của ChatAlpaca.
- ChatAlpaca được sinh bằng ChatGPT (gpt-3.5-turbo), dựa trên dữ liệu Stanford Alpaca (`tatsu-lab/alpaca`).
- Stanford Alpaca ghi giấy phép **CC BY-NC 4.0** (cấm dùng thương mại) và được sinh bằng model của OpenAI (text-davinci-003). Điều khoản của OpenAI hạn chế dùng kết quả sinh ra để làm model cạnh tranh.
- Vì vậy, dù card ghi Apache-2.0, dữ liệu gốc chặt hơn. Kết luận: chỉ dùng cho mục đích cá nhân hoặc nghiên cứu; không bán model hay dịch vụ đã train bằng preset này.

## `vietnamese_aya`: Apache-2.0, do người viết (M26)
- Dataset: `CohereLabs/aya_dataset`, commit `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, không phải chấp nhận điều khoản trước khi tải.
- Card ghi: "This dataset can be used for any purpose, whether academic or commercial, under the terms of the Apache 2.0 License."
- Nguồn gốc (card, mục Provenance): người tình nguyện viết trên Aya Annotation Platform, có bước kiểm tra chất lượng. Có 2 loại dòng:
  - `original-annotations`: câu hỏi và câu trả lời mới hoàn toàn do người viết;
  - `re-annotations`: người sửa lại câu hỏi, câu trả lời sinh tự động từ các dataset NLP mở khác.
- Preset chỉ lấy `original-annotations` tiếng Việt (`language_code` `vie`), vì nguồn gốc rõ nhất; bỏ `re-annotations`, vì phần gốc đến từ dataset khác. Ở commit trên có 8.676 dòng tiếng Việt: 4.853 dòng `original-annotations`, 3.823 dòng `re-annotations` (đếm bằng cách chỉ đọc 2 cột `language_code` và `annotation_type` của file parquet ngày 30/9).
- Card cảnh báo: có thể còn nội dung xúc phạm, dòng gán nhầm ngôn ngữ, hoặc không đúng dạng chỉ dẫn. Bộ lọc chất lượng của repo (ngôn ngữ, độ dài, lặp, gần trùng) vẫn chạy như các preset khác.
- Một số dòng yêu cầu "tiếp tục đoạn văn" có thể chứa đoạn văn người viết chép từ nơi khác. Card không nói tới việc này; giấy phép Apache-2.0 là do Cohere Labs công bố cho cả dataset.
- Nếu công khai model hoặc dữ liệu đã train, nên ghi: "Dữ liệu tiếng Việt: CohereLabs/aya_dataset (Apache-2.0)".

## Việc chủ repo cần làm
1. Đọc bảng tóm tắt và quyết định có chấp nhận điều kiện của từng preset không.
2. Chỉ dùng cá nhân: có thể dùng cả 3 preset (nhớ ghi nguồn khi công khai model train bằng `code`).
3. Muốn dùng thương mại: bỏ preset `vietnamese` khỏi lệnh trộn `--preset`, dùng `vietnamese_aya` (M26) thay thế.
