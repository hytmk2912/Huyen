"""Một lệnh kiểm tra trước khi đánh dấu mốc Xong và tự gộp PR: `python -m local_ai.check`.

Chạy lần lượt:
1. toàn bộ test (`unittest` trong thư mục `tests/`);
2. `compileall` thư mục `local_ai`;
3. `secret-scan` (quét khóa, mật khẩu bị lộ).

Mã thoát khác 0 nếu có test lỗi, compileall lỗi, secret-scan có phát hiện, **hoặc có test bị bỏ qua (skip) vì thiếu thư viện**
(torch, transformers, trl, peft, datasets, tokenizers, nbformat): khi đó các test train thật trên CPU không chạy, nên "xanh" là
không đủ tin. Cài thư viện theo README (torch bản CPU) rồi chạy lại; chỉ khi thật sự không cài được mới dùng `--allow-skip`.
"""
from __future__ import annotations

import argparse
import compileall
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from typing import Iterable

from local_ai.data.secrets import scan_secrets

ROOT = Path(__file__).resolve().parent.parent
LIBRARIES = ("torch", "transformers", "trl", "peft", "datasets", "tokenizers", "nbformat")
INSTALL_HINT = ("Cài thư viện theo README (mục \"Cài đặt và kiểm tra\"): máy không có GPU thì cài torch bản CPU trước "
                "(python -m pip install torch --index-url https://download.pytorch.org/whl/cpu), rồi "
                "python -m pip install transformers trl peft datasets tokenizers nbformat; sau đó chạy lại lệnh này.")


def missing_libraries() -> list[str]:
    return [name for name in LIBRARIES if importlib.util.find_spec(name) is None]


def library_skips(skipped: Iterable[tuple[object, str]]) -> list[tuple[str, str]]:
    """Các test bị bỏ qua vì thiếu thư viện: lý do skip nhắc tới "thiếu" hoặc tên một thư viện trong LIBRARIES."""
    found = []
    for test, reason in skipped:
        text = str(reason)
        if "thiếu" in text.lower() or any(name in text for name in LIBRARIES): found.append((str(test), text))
    return found


def run_tests(test_dir: Path, verbose: bool) -> unittest.TestResult:
    suite = unittest.TestLoader().discover(str(test_dir), top_level_dir=str(test_dir))  # loader mới mỗi lần, như `python -m unittest discover -s tests`
    return unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)


def run_checks(test_dir: Path = ROOT / "tests", allow_skip: bool = False, verbose: bool = False) -> int:
    """Chạy 3 bước kiểm tra, in tóm tắt tiếng Việt; trả về mã thoát (0 là xanh)."""
    problems = []
    result = run_tests(test_dir, verbose)
    skips = library_skips(result.skipped)
    print(f"\n1. Test: {result.testsRun} test, {len(result.failures)} sai, {len(result.errors)} lỗi, {len(result.skipped)} bị bỏ qua"
          + (f" ({len(skips)} vì thiếu thư viện)" if skips else ""))
    if not result.wasSuccessful(): problems.append("có test sai hoặc lỗi")
    if skips:
        for name, reason in skips: print(f"   - bỏ qua {name}: {reason}")
        if not allow_skip: problems.append("có test bị bỏ qua vì thiếu thư viện")
    compiled = compileall.compile_dir(str(ROOT / "local_ai"), quiet=1)
    print("2. compileall local_ai: " + ("sạch" if compiled else "LỖI"))
    if not compiled: problems.append("compileall lỗi")
    findings = scan_secrets(ROOT)
    print("3. secret-scan: " + ("không có phát hiện" if not findings else "CÓ PHÁT HIỆN: " + ", ".join(findings)))
    if findings: problems.append("secret-scan có phát hiện")
    missing = missing_libraries()
    if missing: print("Thiếu thư viện: " + ", ".join(missing))
    if problems:
        print("\nKẾT QUẢ: CHƯA XANH (" + "; ".join(problems) + "). Không đánh dấu mốc Xong, không gộp PR.")
        if skips and not allow_skip: print(INSTALL_HINT)
        return 1
    print("\nKẾT QUẢ: XANH" + (" (có --allow-skip: một số test bị bỏ qua vì thiếu thư viện)" if skips else "") + ".")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.check", description="Chạy toàn bộ test, compileall và secret-scan; test bị bỏ qua vì thiếu thư viện cũng tính là chưa xanh.")
    parser.add_argument("--allow-skip", action="store_true", help="Cho phép test bị bỏ qua vì thiếu thư viện (chỉ dùng khi thật sự không cài được thư viện)")
    parser.add_argument("--tests", default=str(ROOT / "tests"), help="Thư mục chứa test (mặc định tests/ của repo)")
    parser.add_argument("--verbose", "-v", action="store_true", help="In tên từng test khi chạy")
    args = parser.parse_args(argv)
    os.chdir(ROOT)  # test và các đường dẫn cấu hình tính từ thư mục gốc repo
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    return run_checks(Path(args.tests), allow_skip=args.allow_skip, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
