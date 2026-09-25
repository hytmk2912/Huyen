"""Làm việc với Hugging Face Hub khi train (dùng trên Colab): tải checkpoint để train tiếp, đẩy adapter lên repo riêng tư.

Token chỉ đọc từ biến môi trường `HF_TOKEN` (trên Colab: lấy từ Colab Secrets rồi đặt vào biến môi trường).
Thư viện `huggingface_hub` chỉ được import khi thật sự gọi tới Hub.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
LAST_CHECKPOINT = "last-checkpoint"  # thư mục Trainer đẩy lên Hub khi hub_strategy="checkpoint"
DOWNLOAD_DIR = "_hub"  # Trainer bỏ qua mục bắt đầu bằng "_" khi đẩy output_dir lên Hub, nên checkpoint tải về không bị đẩy ngược lên


def check_repo_id(repo_id: str | None) -> str:
    if not repo_id or not REPO_ID.match(repo_id): raise ValueError(f"Tên repo Hugging Face phải có dạng tên-người-dùng/tên-repo, không phải '{repo_id}'")
    return repo_id


def downloaded_checkpoint(output_dir: str | Path) -> Path | None:
    """Checkpoint đã tải từ Hub ở lần chạy trước (`output_dir/_hub/last-checkpoint`), nếu đủ file để chạy tiếp."""
    target = Path(output_dir) / DOWNLOAD_DIR / LAST_CHECKPOINT
    return target if (target / "trainer_state.json").is_file() else None


def download_last_checkpoint(repo_id: str, output_dir: str | Path) -> Path | None:
    """Tải thư mục `last-checkpoint` từ repo về `output_dir/_hub/`. Repo hoặc checkpoint chưa có (lần chạy đầu) thì trả None.

    Lỗi khác (mất mạng, token sai quyền...) vẫn được báo ra, để không lặng lẽ train lại từ đầu rồi ghi đè checkpoint cũ trên Hub.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

    check_repo_id(repo_id)
    try:
        snapshot_download(repo_id=repo_id, allow_patterns=[f"{LAST_CHECKPOINT}/*"], local_dir=str(Path(output_dir) / DOWNLOAD_DIR))
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError):
        return None
    target = downloaded_checkpoint(output_dir)
    if target is None: shutil.rmtree(Path(output_dir) / DOWNLOAD_DIR / LAST_CHECKPOINT, ignore_errors=True)  # tải thiếu thì bỏ, train lại từ đầu thay vì chạy tiếp từ checkpoint hỏng
    return target


def push_adapter(repo_id: str, adapter_dir: str | Path, private: bool = True, dry_run: bool = False) -> dict[str, object]:
    """Tạo repo (mặc định riêng tư) nếu chưa có rồi đẩy thư mục adapter lên. `dry_run` chỉ kiểm tra tham số, không gọi mạng."""
    check_repo_id(repo_id)
    folder = Path(adapter_dir)
    plan = {"repo_id": repo_id, "adapter_dir": str(folder), "private": private, "url": f"https://huggingface.co/{repo_id}"}
    if dry_run: return {"status": "dry-run", **plan}
    if not (folder / "adapter_config.json").is_file(): raise FileNotFoundError(f"Không thấy adapter trong {folder} (thiếu adapter_config.json); hãy train xong trước")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), commit_message="Đẩy adapter LoRA sau khi train")
    return {"status": "pushed", **plan}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.training.hub", description="Đẩy adapter đã train lên Hugging Face Hub (token đọc từ biến môi trường HF_TOKEN).")
    commands = parser.add_subparsers(dest="command", required=True)
    push = commands.add_parser("push-adapter", help="Đẩy thư mục adapter lên repo Hugging Face (mặc định riêng tư)")
    push.add_argument("--repo", required=True, help="Tên repo dạng tên-người-dùng/tên-repo")
    push.add_argument("--adapter-dir", required=True, help="Thư mục adapter (có adapter_config.json)")
    push.add_argument("--public", action="store_true", help="Tạo repo công khai thay vì riêng tư")
    push.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra tham số và in kế hoạch, không gọi mạng")
    args = parser.parse_args(argv)
    print(json.dumps(push_adapter(args.repo, args.adapter_dir, private=not args.public, dry_run=args.dry_run), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
