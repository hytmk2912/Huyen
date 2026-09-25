# Train trên Colab bằng iPhone

Hướng dẫn chạy notebook `notebooks/train_colab.ipynb` trên GPU T4 miễn phí của Google Colab, chỉ cần iPhone (Safari). Notebook sẽ:
1. lấy 2000 dòng dữ liệu;
2. chấm model gốc;
3. train QLoRA;
4. chấm lại, in bảng so sánh trước/sau;
5. đẩy adapter lên repo Hugging Face riêng tư của bạn.

**Lưu ý:** notebook **chưa chạy thử trên Colab thật**, vì môi trường phát triển của repo không có GPU. Nếu gặp lỗi không có trong mục "Lỗi hay gặp", hãy chụp màn hình ô báo lỗi đầu tiên và gửi lại.

Link notebook (dùng được sau khi PR tuần 2 được gộp vào `main`): <https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/train_colab.ipynb>

## Chọn model và thời gian ước tính
| Model | Train | Chấm mỗi lần (tối đa) | Tổng, chưa tính cài đặt và tải model |
| --- | --- | --- | --- |
| `smoke` (Qwen2.5-0.5B), nên chạy trước | khoảng 13 phút | khoảng 4 phút | khoảng 21 phút |
| `light` (Qwen3-4B) | khoảng 75 phút | khoảng 17 phút | khoảng 109 phút |

- Cài thư viện và tải model thường mất thêm 5–10 phút.
- Số trong bảng tính bằng `python -m local_ai.training.estimate` (2000 dòng, GPU T4). Đây là ước lượng thô, chưa đo trên T4 thật.
- Khi chạy, Bước 6 của notebook in ước tính theo đúng dữ liệu vừa lấy.
- `light` chạy lâu, nên Colab miễn phí có thể ngắt giữa chừng. Khi đó xem mục "Khi Colab ngắt giữa chừng": việc train tự chạy tiếp, không mất công.

## Chuẩn bị (làm một lần)
1. **Token Hugging Face:**
   1. đăng nhập <https://huggingface.co>;
   2. chạm ảnh đại diện → **Settings** → **Access Tokens** → **Create new token**;
   3. chọn loại **Write**, đặt tên tùy ý (ví dụ `colab`), rồi **Copy**.

   Không dán token vào notebook, tin nhắn hay file trong repo.
2. **Tài khoản Google** để dùng Colab.
3. **Giữ màn hình sáng khi chạy lâu** (nhất là `light`): cắm sạc, rồi vào **Cài đặt → Màn hình & Độ sáng → Khóa tự động → Không bao giờ**. Nhớ bật lại sau khi xong. Màn hình khóa thì Safari có thể ngừng kết nối, và Colab dễ ngắt hơn.

## Bước 1: mở notebook
1. Mở link notebook ở trên bằng Safari. Cách khác: mở `README.md` của repo trên GitHub, chạm nút **Open in Colab**.
2. Đăng nhập Google nếu được hỏi.
3. Nên chuyển sang giao diện máy tính để thấy đủ menu và cột bên trái: chạm **aA** ở thanh địa chỉ → **Yêu cầu trang web cho máy tính**. Dùng Chrome thì chạm **⋯** → **Yêu cầu trang web cho máy tính**.

Nếu tài khoản Google của bạn để tiếng Việt, tên menu trong Colab có thể hiện bằng tiếng Việt (ví dụ **Runtime** thành "Thời gian chạy").

## Bước 2: chọn GPU T4
Menu **Runtime** → **Change runtime type** → chọn **T4 GPU** → **Save**. Notebook đã đặt sẵn T4, thường chỉ cần kiểm tra lại.

## Bước 3: thêm HF_TOKEN vào Secrets
1. Ở cột bên trái, chạm biểu tượng chìa khóa 🔑 (**Secrets**).
2. Chạm **Add new secret**.
3. **Name**: `HF_TOKEN`; **Value**: dán token vừa copy.
4. Bật công tắc **Notebook access**.

Secret được lưu trong tài khoản Google của bạn, không nằm trong notebook hay repo. Lần sau mở lại notebook không cần thêm lại.

## Bước 4: chọn model
Trong ô **Bước 1** của notebook, chạm ô chọn ở dòng `MODEL` → chọn `smoke` hoặc `light`. Lần đầu nên chọn `smoke`.

## Bước 5: Run all
1. Menu **Runtime** → **Run all**.
2. Có thể có 2 hộp thoại hiện ra:
   - "Warning: This notebook was not authored by Google" → chạm **Run anyway**;
   - hỏi quyền cho notebook đọc secret `HF_TOKEN` → chạm **Grant access**.
3. Ô đang chạy có vòng xoay; ô chạy xong có dấu ✓. Bước 6 in ước tính thời gian; Bước 8 (train) là bước lâu nhất.

## Bước 6: xem kết quả
- **Bảng so sánh điểm** ở **Bước 10**. Hãy chụp màn hình, vì mọi file trong Colab (thư mục `.runs/`) mất khi Colab tắt.
- **Bảng số đo thật** ở **Bước 12**: thời gian train, VRAM và tốc độ chấm, so với ước tính. Hãy chụp màn hình gửi lại để sửa ước tính cho đúng. Bảng này cũng được lưu ở `so_do/<model>.json` trong repo riêng tư.
- **Adapter và checkpoint** nằm ở repo riêng tư `https://huggingface.co/<tên-bạn>/huyen-<model>-qlora`, ví dụ `huyen-smoke-qlora`. Chỉ tài khoản của bạn xem được. Thư mục `last-checkpoint` trong đó dùng để train tiếp.

## Khi Colab ngắt giữa chừng
Trong lúc train, checkpoint mới nhất được đẩy lên repo riêng tư: `light` đẩy sau mỗi 10 bước, `smoke` sau mỗi 25 bước. Khi Colab ngắt ("Runtime disconnected"):
1. mở lại link notebook, kiểm tra vẫn là T4 (Bước 2);
2. giữ nguyên model đã chọn ở ô Bước 1;
3. chạm **Run all**.

Bước 6 sẽ báo "Đã train N/125 bước", rồi Bước 8 tự tải checkpoint về và train tiếp. Chỉ mất phần việc làm sau checkpoint cuối cùng.

Bước 7 không chấm lại model gốc: lần đầu chấm xong, báo cáo đã được lưu vào repo riêng tư (`eval/truoc/`), nên lần chạy lại tải về dùng luôn. Chỉ chấm lại khi bạn đổi model hoặc cài đặt chấm.

## Lỗi hay gặp
| Thấy gì | Cách xử lý |
| --- | --- |
| "Chưa có GPU" | Làm lại Bước 2 (chọn T4), rồi **Run all**. |
| Không kết nối được GPU (ví dụ "Cannot connect to GPU backend") | Hết lượt GPU miễn phí trong ngày. Đợi vài giờ hoặc sang hôm sau rồi thử lại. |
| "Chưa đọc được HF_TOKEN" | Làm lại Bước 3; nhớ bật **Notebook access**. |
| Lỗi 401 hoặc 403 khi đẩy lên Hugging Face | Token chưa có quyền **Write**: tạo token mới loại Write, rồi sửa giá trị secret `HF_TOKEN`. |
| "CUDA out of memory" (hết bộ nhớ GPU) | Chọn `smoke`. Muốn giữ `light` thì phải giảm `max_length` trong `configs/training/colab_light.json` (sửa file trong repo). |
| Ô báo đỏ "Bước N lỗi (mã thoát ...)" | Lệnh của bước đó lỗi nên **Run all** dừng ngay ở ô này; các ô sau chưa chạy. Đọc thông báo ngay phía trên dòng đỏ, sửa xong thì chạy lại từ ô đó (menu Runtime → Run after). |
| pip in dòng "ERROR: pip's dependency resolver ..." khi cài thư viện | Thường chỉ là xung đột với gói có sẵn của Colab. Nếu các ô sau vẫn chạy được thì có thể bỏ qua. |
| Colab ngắt giữa chừng | Xem mục "Khi Colab ngắt giữa chừng". |
