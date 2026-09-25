"""TerminalTool: công cụ chạy lệnh cho agent, dựng trên `executor.run_command` (không qua shell).

Tắt mặc định. Chỉ chạy lệnh có trong allowlist (`configs/tools/terminal.json`), tham số truyền dạng list, có timeout,
mọi lệnh (kể cả lệnh bị từ chối) được ghi vào file log JSONL. Các lớp chặn, theo thứ tự:
1. công cụ đang tắt;
2. ký tự điều khiển shell (`;` `&` `|` `$` backtick `<` `>` xuống dòng) ở bất kỳ đâu;
3. tên lệnh phải là tên trần có trong allowlist (không nhận đường dẫn như `/bin/ls`);
4. lệnh con (`git status`), bộ tham số cố định (`python --version`), tham số bị cấm (`find -exec`), số tham số tối đa;
5. mọi đường dẫn phải nằm trong thư mục làm việc (chặn `/etc/passwd`, `../..`).
"""
from __future__ import annotations

import json
import shlex
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from local_ai.runtime.executor import CommandResult, run_command

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "tools" / "terminal.json"
FORBIDDEN_CHARACTERS = {";": "dấu ;", "&": "dấu &", "|": "dấu |", "$": "dấu $", "`": "dấu backtick", "<": "dấu <", ">": "dấu >", "\n": "xuống dòng", "\r": "xuống dòng", "\x00": "ký tự NUL"}


class CommandRejected(PermissionError):
    """Lệnh bị từ chối trước khi chạy (công cụ tắt, ngoài allowlist, có ký tự shell, đường dẫn ra ngoài...)."""


class CommandFailed(RuntimeError):
    """Lệnh đã chạy nhưng thất bại (mã thoát khác 0 hoặc quá thời gian)."""


@dataclass(frozen=True)
class CommandRule:
    subcommands: tuple[str, ...] = ()
    exact_args: tuple[tuple[str, ...], ...] | None = None
    deny_options: tuple[str, ...] = ()
    max_args: int = 16


@dataclass(frozen=True)
class TerminalConfig:
    enabled: bool = False
    workspace: Path = ROOT / ".runs" / "terminal" / "workspace"
    log_path: Path = ROOT / ".runs" / "terminal" / "log.jsonl"
    timeout_s: float = 30
    max_output: int = 12_000
    allowed_commands: dict[str, CommandRule] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_CONFIG) -> "TerminalConfig":
        raw = {key: value for key, value in json.loads(Path(path).read_text(encoding="utf-8")).items() if not key.startswith("_")}
        rules = {rule["name"]: CommandRule(subcommands=tuple(rule.get("subcommands", ())), exact_args=tuple(tuple(args) for args in rule["exact_args"]) if "exact_args" in rule else None,
                                           deny_options=tuple(rule.get("deny_options", ())), max_args=rule.get("max_args", 16)) for rule in raw.pop("allowed_commands", [])}
        for key in ("workspace", "log_path"):
            if key in raw: raw[key] = Path(raw[key]) if Path(raw[key]).is_absolute() else ROOT / raw[key]
        config = cls(**raw, allowed_commands=rules)
        if config.timeout_s <= 0: raise ValueError("timeout_s phải lớn hơn 0 giây")
        return config


def _denied(arg: str, option: str) -> bool:
    """Tham số `arg` có dùng tùy chọn bị cấm `option` không.

    - Tùy chọn dài (`--set`) và kiểu find (`-exec`): so theo tiền tố, nên `--set=...` và `-exec+` đều bị chặn.
    - Tùy chọn ngắn một chữ (`-s`): chặn cả khi viết gộp, ví dụ `date -us2020-01-01` nghĩa là `-u -s 2020-01-01`.
    """
    if len(option) == 2 and option.startswith("-") and option != "--":
        return arg.startswith("-") and not arg.startswith("--") and option[1] in arg[1:]
    return arg.startswith(option)


class TerminalTool:
    """Gọi như một hàm (`tool(command="ls -la")` hoặc `tool(command=["ls", "-la"])`) để dùng trong ToolRegistry."""

    def __init__(self, config: TerminalConfig | None = None, enabled: bool | None = None):
        self.config = config or TerminalConfig.from_file()
        self.enabled = self.config.enabled if enabled is None else enabled  # bật phải khai báo rõ: trong cấu hình hoặc enabled=True
        self._lock = threading.Lock()

    def check(self, command: str | Sequence[str]) -> list[str]:
        """Trả về argv đã kiểm tra, hoặc ném CommandRejected kèm lý do bằng tiếng Việt."""
        if not self.enabled: raise CommandRejected("TerminalTool đang tắt; muốn bật phải đặt enabled: true trong cấu hình (configs/tools/terminal.json) hoặc tạo công cụ với enabled=True")
        parts = [command] if isinstance(command, str) else list(command)
        if not parts or not all(isinstance(part, str) for part in parts): raise CommandRejected("Lệnh phải là một chuỗi hoặc list các chuỗi, không được rỗng")
        for part in parts:
            for character, label in FORBIDDEN_CHARACTERS.items():
                if character in part: raise CommandRejected(f"Lệnh chứa {label} (ký tự điều khiển shell) nên bị từ chối; công cụ không chạy qua shell")
        try:
            argv = shlex.split(command) if isinstance(command, str) else parts
        except ValueError as error: raise CommandRejected(f"Không tách được lệnh thành các tham số: {error}") from error
        if not argv: raise CommandRejected("Lệnh rỗng")
        name, args = argv[0], argv[1:]
        rule = self.config.allowed_commands.get(name)
        if rule is None or "/" in name: raise CommandRejected(f"Lệnh '{name}' không có trong allowlist; được phép: {', '.join(sorted(self.config.allowed_commands))}")
        if rule.exact_args is not None and tuple(args) not in rule.exact_args: raise CommandRejected(f"Lệnh '{name}' chỉ được chạy với: {' | '.join(' '.join(item) or '(không tham số)' for item in rule.exact_args)}")
        if len(args) > rule.max_args: raise CommandRejected(f"Lệnh '{name}' nhận tối đa {rule.max_args} tham số")
        if rule.subcommands:
            if not args or args[0] not in rule.subcommands: raise CommandRejected(f"Lệnh '{name}' chỉ được dùng với: {', '.join(rule.subcommands)}")
            args = args[1:]
        for arg in args:
            if any(_denied(arg, option) for option in rule.deny_options): raise CommandRejected(f"Tham số '{arg}' bị cấm với lệnh '{name}'")
            if arg.startswith("--"): value = arg.split("=", 1)[1] if "=" in arg else None  # --file=/etc/passwd
            elif arg.startswith("-"): value = arg[2:] if len(arg) > 2 and ("/" in arg[2:] or ".." in arg[2:]) else None  # -f/etc/passwd
            else: value = arg
            if value is not None: self._inside_workspace(value)
        return argv

    def _inside_workspace(self, value: str) -> None:
        workspace = self.config.workspace.resolve()
        target = (workspace / value).resolve()
        if target != workspace and workspace not in target.parents: raise CommandRejected(f"Đường dẫn '{value}' nằm ngoài thư mục làm việc nên bị từ chối")

    def run(self, command: str | Sequence[str]) -> CommandResult:
        """Kiểm tra rồi chạy lệnh; ghi log cả khi bị từ chối."""
        started = time.monotonic()
        try:
            argv = self.check(command)
        except CommandRejected as error:
            self._log({"command": command if isinstance(command, str) else list(command) if isinstance(command, (list, tuple)) else repr(command), "allowed": False, "reason": str(error)})
            raise
        result = run_command(argv, self.config.workspace, timeout=self.config.timeout_s, max_output=self.config.max_output)
        self._log({"command": argv, "allowed": True, "job_id": result.job_id, "status": result.status, "exit_code": result.exit_code, "timed_out": result.timed_out, "duration_s": round(time.monotonic() - started, 3)})
        return result

    def __call__(self, command: str | Sequence[str]) -> str:
        result = self.run(command)
        if result.timed_out: raise CommandFailed(f"Lệnh chạy quá {self.config.timeout_s:g} giây nên đã bị dừng")
        if result.status != "completed": raise CommandFailed(f"Lệnh lỗi (mã thoát {result.exit_code}): {result.stderr.strip()[-500:]}")
        return result.stdout if not result.stderr.strip() else f"{result.stdout}\n[stderr]\n{result.stderr}"

    def _log(self, entry: dict[str, Any]) -> None:
        line = json.dumps({"time": datetime.now(timezone.utc).isoformat(), **entry}, ensure_ascii=False)
        with self._lock:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.log_path.open("a", encoding="utf-8") as handle: handle.write(line + "\n")


TERMINAL_DESCRIPTION = 'Run one allowed shell command (no pipes or redirects) in the working folder, for example ls or cat <file>. Arguments: {"command": "cat notes.txt"}'


def register_terminal(registry: Any, tool: TerminalTool | None = None, name: str = "terminal") -> TerminalTool:
    """Đăng ký TerminalTool vào ToolRegistry của agent. Công cụ đang tắt thì mọi lần gọi đều bị từ chối (agent nhận lỗi, không sập)."""
    tool = tool or TerminalTool()
    registry.register(name, tool, TERMINAL_DESCRIPTION)
    return tool
