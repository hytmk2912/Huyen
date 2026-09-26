# Gộp repo Agent vào Huyen

## Nguồn
- Repo: `huyenytmk2912/agent`. Tên mới là `hytmk2912/Agent`, nhưng phiên làm việc ngày 24/9 không truy cập được tên mới: repo có thể đang để riêng tư, hoặc chưa cấp quyền cho phiên. Tên cũ là repo công khai, đã `git clone` được.
- **Commit gốc đã lấy:** `78a3e25bb4099d52118006f54b4e88aad3313ea2` ("Add job gateway for remote Agent", 1/9/2026). Repo có 4 commit: `769e1d0` (khởi tạo), `b163613` (thư viện), `535f2bf` (runtime v0.1), `78a3e25` (gateway).
- Nội dung repo: `README.md`, `agent_runtime.py` (97 dòng), `gateway.py` (92 dòng), `requirements.txt` (fastapi, uvicorn, pydantic, python-dotenv). **Repo không có test nào.**
- Bản sao nguyên văn được cất ở `archive/agent-goc/` (không import, không chạy).

## Phần đưa vào (mốc M8)
Viết lại bằng thư viện chuẩn của Python, không cần FastAPI. Hành vi giữ như bản gốc, trừ các chỗ sửa để an toàn, được ghi rõ ở cột "Khác bản gốc".

| Bản gốc | Trong Huyen | Khác bản gốc |
| --- | --- | --- |
| `agent_runtime.py`: `execute` chạy lệnh, `TaskResult` (job id, trạng thái, thời điểm bắt đầu/kết thúc, mã thoát, stdout, stderr), cắt output còn 12.000 ký tự cuối, timeout mặc định 120 giây | `local_ai/runtime/executor.py`: `run_command`, `CommandResult` | Lệnh chỉ nhận dạng list, **không bao giờ qua shell**. Quá thời gian thì dừng cả nhóm tiến trình con. Lệnh không tồn tại thì báo lỗi rõ, không ném lỗi ra ngoài. Không tạo thư mục khi import. |
| `agent_runtime.py`: `require_token` (`Bearer <token>`) | `local_ai/runtime/auth.py`: `require_token`, `AuthError` (503 khi chưa cấu hình token, 401 khi token sai) | So sánh bằng `hmac.compare_digest`. Runtime và gateway dùng chung một kiểu `Bearer <token>`; bản gốc mỗi nơi một kiểu. |
| `gateway.py`: `Job`, tạo / xem / nhận (claim) / trả kết quả job, khóa luồng | `local_ai/runtime/jobs.py`: `Job`, `JobQueue`, `JobError` (404, 409, 429) | Giới hạn số job trong bộ nhớ (mặc định 1000; đầy thì bỏ job đã xong cũ nhất). Chỉ job đang chạy mới nhận kết quả. `ok` phải đúng là `true` mới tính "completed". |

Test: `tests/test_m8_runtime.py`. Vì repo Agent không có test, đây là test mới, viết theo đúng hành vi của bản gốc, cộng thêm các chỗ sửa để an toàn.

## Phần làm ở mốc M9 (đã xong)
- **Allowlist lệnh:** bản gốc là hằng `ALLOWED_PREFIXES` trong code (`pwd`, `ls`, `find`, `cat`, `head`, `tail`, `grep`, `git status`, `git log`, `python --version`, `node --version`, `npm --version`, `uname`, `whoami`, `date`). Nay nằm trong `configs/tools/terminal.json`, dùng đúng danh sách này.
- **`TerminalTool`** (`local_ai/runtime/terminal.py`):
  - tắt mặc định;
  - chặn ký tự điều khiển shell, lệnh ngoài allowlist, tham số nguy hiểm và đường dẫn ra ngoài thư mục làm việc;
  - tham số nguy hiểm gồm `find -exec`/`-delete`, `git log --output`, `tail -f`, `grep -R`, `date -s`/`--set` (đổi đồng hồ hệ thống nếu chạy bằng root, ví dụ trên Colab); tham số ngắn bị chặn cả khi viết gộp (`date -us2020-01-01` nghĩa là `-u -s ...`);
  - có timeout; ghi log JSONL từng lệnh, kể cả lệnh bị từ chối;
  - đăng ký vào `ToolRegistry` bằng `register_terminal`.
- **Gateway `/v1/jobs`** (`local_ai/runtime/gateway.py`, viết bằng `http.server`): tắt mặc định, chỉ nghe `127.0.0.1`, bắt buộc token dài ít nhất 16 ký tự đọc từ biến môi trường, giới hạn kích thước nội dung gửi lên.
- **`/v1/execute` qua HTTP của runtime gốc không đưa vào:** chạy lệnh từ xa qua mạng là rủi ro lớn nhất, mà agent đã gọi `TerminalTool` ngay trong tiến trình nên không cần.
- Test: `tests/test_m9_terminal.py`.
- **Giới hạn còn lại:**
  - ~~`git status`/`git log` tự tìm repo git ở thư mục cha, nên đọc được lịch sử của repo chứa thư mục làm việc~~: đã sửa ngày 26/9 (chủ repo thử thật được `git log -p` in cả repo). Nay lệnh chạy với `GIT_CEILING_DIRECTORIES` là thư mục cha của thư mục làm việc và môi trường tối thiểu (không có HF_TOKEN hay khóa; HOME là thư mục tạm). Test: `tests/test_bao_mat_sandbox_terminal.py`;
  - mẫu tìm của `grep` có dấu `/` ở đầu bị coi là đường dẫn và bị từ chối (chặn thừa, nhưng an toàn).

## Phần bỏ
- FastAPI, uvicorn, pydantic, python-dotenv: repo Huyen chỉ dùng thư viện chuẩn cho phần lõi; kiểm tra dữ liệu làm bằng dataclass và code thường.
- Tạo thư mục `./workspace` và đọc biến môi trường ngay khi import: nay thư mục làm việc và token do nơi gọi truyền vào.
- Các phần README gốc ghi là "future" (browser, điều khiển máy tính, VPS/SSH): chưa có code, không đưa vào.

## Rủi ro bảo mật của bản gốc
1. **Chèn lệnh (nghiêm trọng):** `subprocess.run(command, shell=True)`, và allowlist chỉ kiểm tra chuỗi lệnh có bắt đầu bằng một tiền tố cho phép hay không. Vì vậy `ls; rm -rf ~`, `ls && curl … | sh`, `cat $(…)`, dấu backtick hay xuống dòng đều qua được và shell chạy cả phần sau. **Đã sửa ở M8:** chỉ nhận list, không có shell. M9 thêm test chặn từng kiểu chèn lệnh.
2. **Lệnh được phép vẫn nguy hiểm:**
   - `find … -delete` hoặc `find … -exec` xoá hoặc chạy lệnh khác;
   - `git log --output=<file>` ghi đè file;
   - `cat`, `head`, `grep` đọc được file ngoài thư mục làm việc (`~/.ssh/id_rsa`, `/etc/…`). README gốc nói "giới hạn trong workspace", nhưng code không làm.

   M9 cần chặn tham số nguy hiểm và đường dẫn ra ngoài thư mục làm việc.
3. **So sánh token bằng `!=`:** dễ bị tấn công đo thời gian. Gateway so token trần, runtime so `Bearer <token>`: hai nơi không thống nhất. **Đã sửa ở M8.**
4. **Không có log:** README gốc hứa ghi lại mọi lần chạy (job id, lệnh, mã thoát, thời điểm), nhưng code chỉ trả kết quả, không lưu. M9 ghi log từng lệnh.
5. **Cổng mạng:** code không ép địa chỉ nghe. Chạy uvicorn với `--host 0.0.0.0` là lộ ra mạng, không có TLS. M9: tắt mặc định, chỉ `127.0.0.1`.
6. **Gateway:**
   - không giới hạn số job trong RAM, nên có thể bị làm đầy bộ nhớ (**đã sửa ở M8**);
   - ai có token cũng nhận và trả kết quả mọi job;
   - kết quả là JSON tuỳ ý (M8 chỉ nhận object).
7. **Token lấy từ `.env`** (python-dotenv): không có token nào trong repo gốc. Secret-scan của Huyen quét cả `archive/agent-goc/` và không thấy gì.

## Việc chủ repo nên làm sau khi gộp xong
- Archive repo Agent cũ (Settings → Archive this repository), để không ai chạy nhầm bản có lỗ hổng.
- Nếu repo Agent đang để riêng tư dưới tên `hytmk2912/Agent` và có code mới hơn commit `78a3e25`, hãy báo lại, để gộp thêm phần mới.
