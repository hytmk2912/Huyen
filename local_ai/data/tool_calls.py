"""Dòng dữ liệu gọi công cụ tự sinh (M20), để model giữ khả năng trả JSON gọi công cụ sau khi train.

Khi chạy thật trên Colab (M15), `light` tụt tool_use 7/8 → 3/8 sau khi train, vì 2000 dòng train (code, toán, hội thoại
tiếng Việt) không có dòng nào trả lời bằng JSON gọi công cụ. `hf-sft --tool-calls 0.1` trộn thêm khoảng 10% dòng loại này.

Mỗi dòng: câu hỏi liệt kê 3 công cụ (calculator, read_file, search) và yêu cầu trả lời bằng đúng dạng JSON mà nhóm tool_use
của bộ chấm yêu cầu, `{"tool": <tên hoặc null>, "arguments": {...}}`; câu trả lời là JSON đó. Có cả câu không cần công cụ
(`"tool": null`). Để không học thuộc bộ chấm:
- lời dẫn và cách hỏi viết khác bộ chấm (bộ chấm dùng "Available tools: ... Task: ..."), thứ tự công cụ thay đổi;
- số, tên file, chủ đề tìm kiếm và câu không cần công cụ khác bộ chấm; số nào có trong câu tool_use của bộ chấm thì không dùng;
- sau đó `build_dataset` còn chặn gần trùng với bộ chấm bằng MinHash (dòng nào quá giống thì bị loại, ghi trong manifest).

Chỉ dùng thư viện chuẩn và số ngẫu nhiên có seed: không gọi model hay API nào, chạy lại ra đúng các dòng cũ.
"""
from __future__ import annotations

import json
import random
import re
from itertools import permutations
from pathlib import Path
from typing import Any

from local_ai.data.core import load_records

DEFAULT_EVAL = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_v1.jsonl"
TOOLS = {"calculator": "calculator(expression)", "read_file": "read_file(path)", "search": "search(query)"}
JSON_FORMAT = {"en": '{"tool": <name or null>, "arguments": {...}}', "vi": '{"tool": <tên hoặc null>, "arguments": {...}}'}
KINDS = ("calculator", "read_file", "search", None)
SOURCE = {"name": "local_ai/data/tool_calls.py (tự sinh trong repo)", "url": "https://github.com/hytmk2912/Huyen/blob/main/local_ai/data/tool_calls.py"}
LICENSE = {"name": "tự sinh trong repo", "usage": "dữ liệu train tự sinh bằng mẫu câu, không lấy từ dataset ngoài"}

# Lời dẫn: {tools} là danh sách công cụ, {format} là dạng JSON, {task} là việc cần làm. Khác lời dẫn của bộ chấm.
INSTRUCTIONS = {
    "en": ("You can call these tools: {tools}. Decide which one fits the request and answer with a single JSON object {format} and no other text. Request: {task}",
           "Tools you may use: {tools}. Output nothing but one JSON object shaped like {format}; put null in \"tool\" when no tool is needed. User request: {task}",
           "{task}\n\nPick the right tool from this list: {tools}. Respond only with JSON in the format {format}."),
    "vi": ("Bạn có thể gọi các công cụ sau: {tools}. Hãy chọn công cụ phù hợp và chỉ trả lời bằng một đối tượng JSON {format}, không viết gì thêm. Yêu cầu: {task}",
           "Danh sách công cụ: {tools}. Chỉ in ra một JSON có dạng {format}; nếu không cần công cụ thì để \"tool\" là null. Người dùng hỏi: {task}",
           "{task}\n\nChọn đúng công cụ trong danh sách: {tools}. Chỉ trả lời bằng JSON theo mẫu {format}."),
}
# Việc cần tính: (câu hỏi, biểu thức); {a}, {b}, {c} là số, {op} là phép tính.
CALCULATIONS = {
    "en": (("calculate {a} {op} {b}.", "{a} {op} {b}"), ("how much is ({a} + {b}) * {c}?", "({a} + {b}) * {c}"), ("work out {a} * {b} - {c}.", "{a} * {b} - {c}"),
           ("a box holds {a} pens. How many pens are in {b} boxes?", "{a} * {b}"), ("split {a} dollars equally among {c} people.", "{a} / {c}")),
    "vi": (("tính {a} {op} {b}.", "{a} {op} {b}"), ("({a} + {b}) * {c} bằng bao nhiêu?", "({a} + {b}) * {c}"), ("tính giúp {a} * {b} - {c}.", "{a} * {b} - {c}"),
           ("mỗi thùng có {a} chai nước, {b} thùng có tất cả bao nhiêu chai?", "{a} * {b}"), ("chia đều {a} nghìn đồng cho {c} người thì mỗi người được bao nhiêu?", "{a} / {c}")),
}
FILES = {
    "en": (("read {path} and tell me what it says.", ("todo.md", "config/settings.yaml", "logs/server.log", "docs/intro.md", "invoices/march.csv", "report_autumn.txt")),
           ("what is written in {path}?", ("readme_old.txt", "notes/meeting_monday.md", "data/customers.csv", "plan/next_year.txt"))),
    "vi": (("mở file {path} xem trong đó có gì.", ("ke_hoach_tuan.md", "danh_ba.csv", "tai_lieu/huong_dan.txt", "chi_tieu_thang_tam.csv")),
           ("đọc nội dung của {path}.", ("nhat_ky.txt", "hop_dong/mau_hop_dong.md", "diem_thi/lop_chuyen_toan.csv", "cong_thuc/banh_mi.txt"))),
}
SEARCHES = {
    "en": (("search the internet for {topic}.", ("the population of Canada", "the opening hours of the British Museum", "the latest Python release notes", "reviews of electric scooters")),
           ("find online information about {topic}.", ("train times from Paris to Lyon", "the tallest building in Asia", "a recipe for banana bread", "current exchange rate of euro to yen"))),
    "vi": (("tìm trên internet {topic}.", ("giá xăng hôm nay", "lịch thi đấu V-League tuần này", "tỉ giá đô la Mỹ hôm nay", "dự báo bão ở miền Trung")),
           ("tra cứu thông tin về {topic}.", ("giờ mở cửa của Bảo tàng Lịch sử", "điểm chuẩn đại học năm nay", "cách làm phở bò", "tàu hỏa từ Hà Nội đi Huế"))),
}
NO_TOOL = {
    "en": ('the user says "thanks a lot, that helped".', "the user asks you to explain in one sentence what a noun is.", "the user wants a short motivational quote.",
           'the user writes "good night, see you tomorrow".', "the user asks you to suggest a name for a pet cat.", "the user asks what the plural of \"mouse\" is."),
    "vi": ('người dùng nói "cảm ơn bạn nhiều nhé".', "người dùng nhờ giải thích ngắn gọn danh từ là gì.", "người dùng muốn một câu chúc buổi sáng vui vẻ.",
           'người dùng chào "hẹn gặp lại bạn ngày mai".', "người dùng nhờ đặt tên cho một chú mèo con.", "người dùng hỏi từ trái nghĩa của \"cao\" là gì."),
}


def eval_tool_prompts(path: str | Path = DEFAULT_EVAL) -> list[str]:
    """Câu hỏi của nhóm tool_use trong bộ chấm."""
    return [str(record["prompt"]) for record in load_records(path) if record.get("group") == "tool_use"]


def eval_numbers(prompts: list[str]) -> set[int]:
    """Các số có trong câu tool_use của bộ chấm: dòng tự sinh không dùng những số này."""
    return {int(number) for prompt in prompts for number in re.findall(r"\d+", prompt)}


def _tools_line(rng: random.Random) -> str:
    order = rng.choice(list(permutations(TOOLS)))
    return ", ".join(TOOLS[name] for name in order)


def _number(rng: random.Random, banned: set[int], low: int = 13, high: int = 989) -> int:
    while True:
        value = rng.randint(low, high)
        if value not in banned: return value


def _task(kind: str | None, language: str, rng: random.Random, banned: set[int]) -> tuple[str, dict[str, Any]]:
    """(việc cần làm, JSON trả lời) cho một loại công cụ."""
    if kind == "calculator":
        template, expression = rng.choice(CALCULATIONS[language])
        values = {"a": _number(rng, banned), "b": _number(rng, banned), "c": _number(rng, banned, 2, 12), "op": rng.choice("+-*")}
        if "/" in expression: values["a"] = values["c"] * _number(rng, banned, 11, 99)  # chia hết cho gọn
        return template.format(**values), {"tool": "calculator", "arguments": {"expression": expression.format(**values)}}
    if kind == "read_file":
        template, paths = rng.choice(FILES[language]); path = rng.choice(paths)
        return template.format(path=path), {"tool": "read_file", "arguments": {"path": path}}
    if kind == "search":
        template, topics = rng.choice(SEARCHES[language]); topic = rng.choice(topics)
        return template.format(topic=topic), {"tool": "search", "arguments": {"query": topic}}
    return rng.choice(NO_TOOL[language]), {"tool": None, "arguments": {}}


def tool_call_records(count: int, seed: int = 17, version: str = "tool-calls-v1", eval_path: str | Path = DEFAULT_EVAL) -> list[dict[str, Any]]:
    """`count` bản ghi gọi công cụ (schema dữ liệu của repo, có `messages`), chia đều tiếng Việt / tiếng Anh và 4 loại
    (calculator, read_file, search, không cần công cụ). Câu hỏi không lặp nhau; không dùng số có trong câu tool_use của bộ chấm."""
    if count < 0: raise ValueError("Số dòng gọi công cụ phải không âm")
    rng, banned = random.Random(seed), eval_numbers(eval_tool_prompts(eval_path))
    records, seen, attempts = [], set(), 0
    while len(records) < count:
        attempts += 1
        if attempts > 50 * count + 100: raise ValueError(f"Không sinh đủ {count} câu gọi công cụ khác nhau; hãy giảm số dòng")
        position = len(records); language = ("vi", "en")[position % 2]; kind = KINDS[(position // 2) % len(KINDS)]
        task, answer = _task(kind, language, rng, banned)
        prompt = rng.choice(INSTRUCTIONS[language]).format(tools=_tools_line(rng), format=JSON_FORMAT[language], task=task)
        if prompt in seen: continue
        seen.add(prompt)
        reply = json.dumps(answer, ensure_ascii=False)
        records.append({"id": f"tool-call-{version}-{position:05d}", "domain": "tool_use", "task": "sft", "input": prompt, "expected_output": reply,
                        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}], "source": dict(SOURCE), "license": dict(LICENSE),
                        "dataset_version": version, "tools_used": [answer["tool"]] if answer["tool"] else [],
                        "metadata": {"language": language, "synthetic": True, "tool": answer["tool"]}})
    return records


def tool_call_rows(total: int, share: float) -> tuple[int, int]:
    """Chia `total` dòng thành (dòng lấy từ preset, dòng gọi công cụ tự sinh) để dòng tự sinh chiếm khoảng `share`."""
    if not 0 <= share < 1: raise ValueError(f"Tỉ lệ dòng gọi công cụ phải nằm trong [0, 1), không phải {share}")
    synthetic = round(total * share)
    return total - synthetic, synthetic
