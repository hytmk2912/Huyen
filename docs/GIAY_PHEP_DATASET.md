# Giấy phép 3 dataset preset

Ngày đọc: 25/9/2026, trên dataset card (trang giới thiệu dataset) ở Hugging Face. Mỗi preset khóa đúng một commit (`revision` trong `configs/datasets/presets/*.json`), nên nội dung dưới đây ứng với commit đó.

**Lưu ý:** đây là kết quả đọc tài liệu công khai, không phải tư vấn pháp lý. Trạng thái trong preset vẫn để "cần kiểm tra lại trên dataset card": chỉ chủ repo mới quyết định được có chấp nhận các điều kiện dưới đây hay không.

## Tóm tắt
| Preset | Dataset | Giấy phép ghi trên card | Nguồn gốc dữ liệu | Dùng được cho |
| --- | --- | --- | --- | --- |
| `code` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | ODC-By | Hàm Python mẫu lấy từ quy trình MultiPL-T; đề bài và lời giải do StarCoder2-15B tự sinh, lời giải đã chạy thử | Cá nhân và thương mại, **phải ghi nguồn** |
| `reasoning` | `open-r1/OpenR1-Math-220k` | Apache-2.0 | Đề bài từ NuminaMath 1.5; lời giải do DeepSeek R1 sinh; kiểm tra bằng Math Verify, 12% dùng Llama-3.3-70B-Instruct làm giám khảo | Cá nhân và thương mại; nên ghi nguồn |
| `vietnamese` | `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` | Apache-2.0 | Bản dịch tiếng Việt của ChatAlpaca; ChatAlpaca sinh bằng ChatGPT (gpt-3.5-turbo) từ dữ liệu Stanford Alpaca | **Chỉ dùng cá nhân hoặc nghiên cứu, không dùng thương mại** (xem dưới) |

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

## Việc chủ repo cần làm
1. Đọc bảng tóm tắt và quyết định có chấp nhận điều kiện của từng preset không.
2. Chỉ dùng cá nhân: có thể dùng cả 3 preset (nhớ ghi nguồn khi công khai model train bằng `code`).
3. Muốn dùng thương mại: bỏ preset `vietnamese` khỏi lệnh trộn `--preset`, và tìm dataset tiếng Việt khác có nguồn gốc rõ ràng (việc này thuộc đề xuất M20).
