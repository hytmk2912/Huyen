# Kiến trúc nền tảng AI tự động chạy cục bộ

## Phạm vi
Repo này cung cấp hạ tầng điều phối, không phải một model đã huấn luyện, cũng không kết nối trực tiếp với thị trường hay web. Trọng số model, chỉ mục tìm kiếm, dataset và khóa truy cập đều nằm bên ngoài và được chọn qua cấu hình.

## Các tầng

1. **Cấu hình và tái lập**: file cấu hình JSON được nạp thành thiết lập chạy bất biến. `seed_everything` ghi lại và cố định seed ngẫu nhiên của Python; `RunTracker` ghi cấu hình, chỉ số, sự kiện và tham chiếu checkpoint vào thư mục của lượt chạy.
2. **Truy cập model**: `ModelAdapter` là giao thức không phụ thuộc backend. `ScriptedModelAdapter` là adapter tất định dùng khi phát triển cục bộ. `HuggingFaceModelAdapter` chọn lớp nạp theo `kind` của model (`text`: `AutoModelForCausalLM`; `multimodal`: `AutoProcessor` + `AutoModelForMultimodalLM`, nhận được ảnh), nén 4bit/8bit bằng bitsandbytes theo `quantization`, và tự chuyển bf16 sang fp16 trên GPU không hỗ trợ bf16. `local_ai.models.vram` ước tính VRAM từ `params_b` mà không tải model. Có thể thêm adapter cho các engine suy luận cục bộ mà không cần sửa agent.
3. **Định tuyến**: `ModelRouter` chọn model đã cấu hình theo khả năng, có thể đặt model mặc định dự phòng.
4. **Vòng lặp agent**: `AutonomousAgent` chạy yêu cầu → lập kế hoạch → chọn công cụ → thực thi → quan sát → đánh giá → sửa/thử lại → câu trả lời cuối. Bước lập kế hoạch và đánh giá được thiết kế thành giao diện có cấu trúc để có thể thay bằng model hoặc quy tắc tất định.
5. **Công cụ và sandbox**: `ToolRegistry` cung cấp các công cụ đã khai báo. `PythonSandbox` chạy đoạn code được đưa vào trong một thư mục làm việc tạm, có giới hạn thời gian và thu lại đầu ra. Đây chỉ là giao diện ranh giới cách ly, không phải bảo đảm an toàn trước code độc hại; khi chạy thật phải cách ly bằng hệ điều hành/container.
6. **Ngữ cảnh và tri thức**: `MemoryStore` lưu một số lượng giới hạn tin nhắn hội thoại. `Retriever` là giao thức cho nguồn tri thức bên ngoài/RAG, để các thông tin hay thay đổi không phải nằm trong trọng số.
7. **Dữ liệu, huấn luyện và đánh giá**: manifest dataset ghi phiên bản, nguồn và cách chia tập. Kế hoạch huấn luyện hỗ trợ các lượt SFT, tối ưu theo sở thích (preference optimization) và RFT trong tương lai. `local_ai.data.hub` chuyển các dòng dataset Hugging Face sang schema của repo trước bước build chuẩn; `local_ai.training.finetune` là khung SFT tùy chọn (full, LoRA hoặc QLoRA trên model nén 4bit), tự bỏ qua khi thiếu GPU hoặc thư viện. Bộ benchmark dùng chung một hàm đánh giá và chia theo từng nhóm khả năng.

## Điểm mở rộng
- Cài đặt `ModelAdapter.generate` cho một backend cục bộ (ví dụ máy chủ cục bộ hoặc runtime chạy trong tiến trình).
- Khai báo model kèm khả năng trong phần `models` của cấu hình.
- Đăng ký các công cụ đã được kiểm duyệt qua `ToolRegistry`; các giao diện dự kiến gồm lập trình, suy luận, nghiên cứu, giao dịch, truy xuất, chạy lệnh terminal và thao tác file.
- Cài đặt `Retriever.search` cho chỉ mục vector hoặc nguồn dữ liệu trực tiếp.
- `local_ai.training.finetune` là bộ huấn luyện cụ thể đầu tiên đứng sau `TrainingPlan`; preference optimization và RFT vẫn là việc của tương lai.

## An toàn và giới hạn
Quyền dùng công cụ được khai báo rõ ràng, bị giới hạn bởi số vòng lặp và thời gian chờ, và mọi kết quả công cụ đều được quan sát trước quyết định tiếp theo. Bản demo đi kèm chỉ dùng một model tất định và công cụ máy tính. Adapter giao dịch chỉ là giao diện phân tích và không đặt lệnh.
