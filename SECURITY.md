# Bảo mật

Repo này fine-tune LLM và tải model, dataset từ Hugging Face. Hai rủi ro chính: **lộ HF token** và **nạp model chứa mã độc** (file pickle `.bin`/`.pkl` có thể chạy lệnh ngay lúc nạp).

## 1. Secret repo cần và nơi lưu

| Biến | Dùng để | Lưu ở đâu |
| --- | --- | --- |
| `HF_TOKEN` | đọc dataset/model riêng tư; đẩy adapter, báo cáo chấm, số đo lên repo riêng tư | Colab/Kaggle: Secrets (bật Notebook access). Máy riêng: biến môi trường hoặc file `.env` (đã nằm trong `.gitignore`). Nên dùng token fine-grained, chỉ cấp quyền Write cho repo của bạn. |
| `LOCAL_AI_GATEWAY_TOKEN` | token của gateway HTTP local (ít nhất 16 ký tự), chỉ khi bật gateway | biến môi trường |
| Biến ghi ở `api_key_env` của model server local (nếu có) | khóa của server Ollama/llama.cpp/vLLM | biến môi trường |

- Mẫu biến: `.env.example` (giá trị để trống).
- Code chỉ đọc secret qua `local_ai/config/secrets.py`. Thiếu biến bắt buộc thì lệnh dừng và chỉ báo **tên** biến, không có giá trị mặc định.
- Không in, không ghi log giá trị secret. `HF_TOKEN` không được ghi vào `manifest.json`, kế hoạch train hay checkpoint (test: `tests/test_bao_mat_hardening.py`).

## 2. Các lớp bảo vệ có sẵn trong repo

- **Nạp model:** luôn `trust_remote_code=False` và `use_safetensors=True`. Repo hay adapter chỉ có `.bin`/`.pkl` thì bị từ chối; muốn nạp phải chạy lại lệnh với cờ `--allow-pickle-weights` (chỉ dùng khi tin tưởng nguồn). Không nơi nào gọi `pickle.load` hay `torch.load` thiếu `weights_only=True`.
- **Ghim phiên bản:** mọi model Hugging Face trong `configs/models/platform.json` và mọi dataset preset phải ghim `revision` bằng mã commit 40 ký tự; chưa ghim thì lệnh từ chối.
- **Ghi dữ liệu:** `hf-sft` chỉ ghi vào `data/`, thư mục tạm, hoặc thư mục trong biến `LOCAL_AI_DATA_DIR` (chặn `../`, symlink ra ngoài). Tên preset chỉ gồm chữ, số, `_`, `-`.
- **Trước khi commit:** `pre-commit` chạy gitleaks, `detect-private-key`, `check-added-large-files` (cài: `python -m pip install -r requirements-dev.txt && pre-commit install`).
- **GitHub Actions:** `.github/workflows/security.yml` chạy gitleaks (toàn bộ lịch sử) và pip-audit mỗi lần push và pull request; quyền mặc định `contents: read`; action ghim theo mã commit; không dùng `pull_request_target`.
- **Thư viện:** ghim đúng phiên bản trong `requirements.txt`, `requirements-gpu.txt`, `requirements-dev.txt`; bản khóa đầy đủ kèm hash ở `requirements.lock`. Dependabot kiểm tra hằng tuần (`.github/dependabot.yml`).
- **pip-audit (5/10/2026):** `pip-audit -r requirements.lock` và `pip-audit -r requirements-dev.txt -r requirements-gpu.txt`: không có lỗ hổng nào. Chưa có lỗi high/critical nào phải để lại.
- **Checkpoint tải từ Hub để train tiếp:** chỉ tải từ repo riêng tư của chính bạn (`--hub-model-id`). Trainer của transformers nạp trạng thái optimizer bằng `torch.load`; đừng trỏ `--hub-model-id` tới repo của người khác.

## 3. Việc chủ repo tự làm trên GitHub và Hugging Face

- [ ] Bật 2FA cho tài khoản GitHub, dùng **passkey** (Settings → Password and authentication).
- [ ] Thu hồi OAuth app và GitHub App cũ không dùng nữa (Settings → Applications → Authorized OAuth Apps / Installed GitHub Apps).
- [ ] Xoá personal access token cũ (Settings → Developer settings → Personal access tokens) và token Hugging Face cũ (huggingface.co/settings/tokens).
- [ ] Tạo **ruleset** cho nhánh `main` (repo → Settings → Rules → Rulesets → New branch ruleset): chặn force push, chặn xoá nhánh, bắt buộc pull request trước khi gộp, bắt buộc check `security` xanh.
- [ ] Bật Dependabot alerts và Dependabot security updates (repo → Settings → Code security), bật Secret scanning và Push protection.
- [ ] Đổi mật khẩu rsync (user `huyen`) đã lộ trong lịch sử commit `e6bc723` (xem mục 5).
- [ ] Cân nhắc chuyển repo sang Private (khi đó `git clone` trong notebook cần token).

## 4. Khi nghi lộ secret

Làm theo thứ tự, càng sớm càng tốt:

1. **Thu hồi ngay** secret bị lộ: HF token → huggingface.co/settings/tokens (Invalidate/Delete); GitHub token → Developer settings; mật khẩu máy chủ → đổi trên máy chủ.
2. **Tạo secret mới** với quyền tối thiểu, cập nhật vào Colab/Kaggle Secrets hoặc biến môi trường. Không dán vào chat, issue hay notebook.
3. **Kiểm tra dấu vết dùng trái phép:** lịch sử repo trên Hugging Face (commit lạ, repo mới tạo), Security log của GitHub (Settings → Security log), log máy chủ.
4. **Xoá secret khỏi code** bằng một commit mới; chạy `gitleaks detect --source . --log-opts="--all"` để chắc không còn sót.
5. **Viết lại lịch sử git** (chỉ sau khi đã thu hồi; cần chủ repo duyệt vì phải force push): dùng `git filter-repo`, rồi báo mọi người clone lại. Lịch sử cũ có thể đã bị sao chép, nên bước 1 mới là bước bắt buộc.

## 5. Secret đã lộ trong lịch sử (kiểm tra 5/10/2026)

- `cpu_hub.sh` và `gpu_8h.sh` ở commit `e6bc723` (và các commit cũ chứa 2 file này): mật khẩu rsync daemon (user `huyen`) ghi làm giá trị mặc định của biến `SYNC_PASS`. gitleaks không bắt được mẫu này; tìm ra bằng grep. **Coi như đã lộ: phải đổi mật khẩu** trên máy chủ cũ (nếu máy còn chạy).
- Không thấy HF token, khóa API, khóa riêng (private key) hay GitHub token nào trong lịch sử mọi nhánh.
- Lệnh xoá khỏi lịch sử (chờ chủ repo duyệt, chưa chạy; phải force push mọi nhánh):

  ```bash
  pip install git-filter-repo
  git clone --mirror https://github.com/hytmk2912/Huyen.git huyen-mirror && cd huyen-mirror
  git filter-repo --invert-paths --path cpu_hub.sh --path gpu_8h.sh
  git push --force --mirror origin
  ```
