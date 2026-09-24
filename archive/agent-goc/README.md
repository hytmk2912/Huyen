# Code gốc của repo Agent (đã cất, không dùng)

Bản sao nguyên văn của repo `huyenytmk2912/agent` (tên mới `hytmk2912/Agent`) tại commit `78a3e25bb4099d52118006f54b4e88aad3313ea2`, ngày 1/9/2026. Cất ở đây để không mất code khi repo Agent cũ được archive.

- `agent_runtime.py`: server FastAPI chạy lệnh (`POST /v1/execute`).
- `gateway.py`: hàng đợi job qua HTTP.
- `requirements.txt`, `README_goc.md`: thư viện và README gốc (tiếng Anh).

**Không chạy code này.** `agent_runtime.py` chạy lệnh bằng `shell=True` với allowlist theo tiền tố chuỗi, nên bị chèn lệnh (ví dụ `ls; rm -rf ~`). Phần dùng được đã viết lại an toàn trong `local_ai/runtime/`. Chi tiết: `docs/GOP_AGENT.md`.
