"""Chạy lệnh từ notebook Colab sao cho lệnh lỗi thì dừng Run all (mốc M16).

Trong Colab, lệnh `!python ...` lỗi không làm ô báo lỗi, nên Run all vẫn chạy tiếp các ô sau và lỗi dây chuyền
(ví dụ bước dữ liệu hỏng thì bước train cũng hỏng, rất khó tìm lỗi gốc trên điện thoại). `run` chạy lệnh dạng list,
không qua shell, in output ngay khi có. Mã thoát khác 0 thì ném `StepFailed` với thông báo tiếng Việt, nên ô đó báo đỏ
và Colab dừng Run all ở đúng chỗ.
"""
from __future__ import annotations

import codecs
import os
import shlex
import subprocess
import sys
from typing import TextIO


class StepFailed(RuntimeError):
    """Một bước của notebook lỗi; Colab dừng Run all ở ô này."""


def command_argv(command: str | list[str]) -> list[str]:
    """Tách lệnh thành list. `python`/`python3` chạy bằng đúng Python của notebook; `pip` chạy bằng `python -m pip`."""
    argv = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
    if not argv: raise ValueError("Lệnh rỗng")
    if argv[0] in ("python", "python3"): return [sys.executable, *argv[1:]]
    if argv[0] == "pip": return [sys.executable, "-m", "pip", *argv[1:]]
    return argv


def run(command: str | list[str], step: str, env: dict[str, str] | None = None, quiet: bool = False, output: TextIO | None = None) -> None:
    """Chạy một lệnh của bước `step`. `quiet` thì chỉ in output khi lệnh lỗi (ví dụ apt-get). Lỗi thì ném StepFailed."""
    stream = output or sys.stdout
    argv = command_argv(command)
    # PYTHONUNBUFFERED: tiến trình Python con in ngay từng dòng (log train), không đợi đầy bộ đệm mới in.
    environ = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    try:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environ)
    except FileNotFoundError as error:
        raise StepFailed(f"{step} lỗi: không tìm thấy lệnh '{argv[0]}'. Hãy chạy lại các bước cài đặt phía trên.") from error
    decoder, captured = codecs.getincrementaldecoder("utf-8")(errors="replace"), []

    def emit(text: str) -> None:
        if not text: return
        if quiet: captured.append(text)
        else: stream.write(text); stream.flush()

    try:
        with process.stdout:  # đóng pipe khi xong (kể cả khi lỗi), không để sót file mở
            while chunk := process.stdout.read1(4096):  # đọc mẩu nào in mẩu đó; giữ nguyên \r để thanh tiến trình hiện đúng
                emit(decoder.decode(chunk))
            emit(decoder.decode(b"", final=True))  # in nốt phần còn dở trong decoder (ví dụ byte UTF-8 cuối bị cắt ngang)
        code = process.wait()
    except KeyboardInterrupt:  # bấm dừng ô thì dừng cả lệnh con, không để nó chạy ngầm
        process.terminate(); process.wait()
        raise
    if code != 0:
        if quiet: stream.write("".join(captured)[-4000:]); stream.flush()
        raise StepFailed(f"{step} lỗi (mã thoát {code}); xem thông báo lỗi ngay phía trên. Các ô sau chưa chạy. Sửa xong thì chạy lại từ ô này (menu Runtime → Run after, hoặc Run all).")
