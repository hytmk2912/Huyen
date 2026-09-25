"""Bộ lọc chất lượng dữ liệu SFT, chỉ dùng thư viện chuẩn: ngôn ngữ (ưu tiên tiếng Việt), độ dài, lặp từ/câu, gần trùng (MinHash).

Các bộ lọc chạy sau bước kiểm tra schema và loại trùng tuyệt đối. Bản ghi bị loại được ghi vào `rejected.jsonl` kèm
tên bộ lọc và lý do; số dòng, ngôn ngữ và độ dài trước/sau lọc ghi vào mục `quality` của `manifest.json`.
Ngưỡng mặc định nằm trong `configs/datasets/quality.json`.

Ưu tiên tiếng Việt nghĩa là:
- nhận ra tiếng Việt trước tiên, dựa vào chữ có dấu, nên bản ghi trộn tiếng Việt với code hay tiếng Anh vẫn tính là tiếng Việt;
- nguồn khai báo `language: "vi"` (preset vietnamese) thì nội dung phải là tiếng Việt có dấu; ngược lại, dòng tiếng Việt có dấu
  trong nguồn khai báo ngôn ngữ khác (ví dụ preset code) vẫn được giữ;
- tiếng Việt không dấu bị loại;
- manifest ghi tỉ lệ tiếng Việt trước và sau lọc.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from local_ai.data.core import normalize_text, sft_messages

DEFAULT_QUALITY = Path(__file__).resolve().parents[2] / "configs" / "datasets" / "quality.json"
VIETNAMESE_LETTERS = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
# Từ rất hay gặp; tiếng Việt không dấu chỉ lấy những từ ít khi là từ tiếng Anh.
ENGLISH_WORDS = set("the of and to is in that for it with as be on are this by an we can if not or from at which you have will what how was".split())
VIETNAMESE_ASCII_WORDS = set("khong cua nhung duoc trong nguoi voi mot nay cac nhu thi cho minh ban toi lam nhieu cung chung dung phai nhat".split())
CODE = re.compile(r"```.*?(?:```|$)", re.S)
LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
WORD = re.compile(r"\w+", re.U)
SENTENCE = re.compile(r"(?<=[.!?。])\s+|\n+")


def conversation(record: dict[str, Any]) -> tuple[str, str]:
    """(nội dung các lượt user, nội dung các lượt assistant) của hội thoại sẽ ghi vào sft.jsonl."""
    messages = sft_messages(record)
    pick = lambda role: "\n".join(str(message["content"]) for message in messages if message["role"] == role)
    return pick("user"), pick("assistant")


def prose(text: str) -> str:
    """Phần chữ để xét ngôn ngữ và độ lặp: bỏ khối code ```...``` và lệnh LaTeX (\\frac, \\cdot...), vì chúng tự nhiên lặp nhiều
    và không mang ngôn ngữ. Code viết trong dòng (`...`) giữ lại vì ngắn."""
    return LATEX_COMMAND.sub(" ", CODE.sub(" ", text))


def detect_language(text: str) -> str:
    """Trả về "vi", "vi-khong-dau", "en", "other" (chữ không phải Latin chiếm nhiều, ví dụ tiếng Trung) hoặc "unknown" (quá ít chữ)."""
    letters = [char for char in unicodedata.normalize("NFC", text).casefold() if char.isalpha()]
    if len(letters) < 20: return "unknown"
    latin = [char for char in letters if char.isascii() or unicodedata.name(char, "").startswith("LATIN")]
    if len(latin) < 0.7 * len(letters): return "other"
    if sum(char in VIETNAMESE_LETTERS for char in latin) >= 0.05 * len(latin): return "vi"
    words = WORD.findall(text.casefold())
    if sum(word in VIETNAMESE_ASCII_WORDS for word in words) >= 0.08 * len(words): return "vi-khong-dau"
    return "en" if sum(word in ENGLISH_WORDS for word in words) >= 0.05 * len(words) else "unknown"


def record_language(record: dict[str, Any]) -> str:
    prompt, answer = conversation(record)
    return detect_language(prose(f"{prompt}\n{answer}"))


@dataclass(frozen=True)
class LanguageFilter:
    enabled: bool = True
    allowed: tuple[str, ...] = ("vi", "en", "unknown")  # ngôn ngữ được giữ; "unknown" là bản ghi gần như chỉ có code hoặc số
    match_source_language: bool = True  # nguồn khai báo metadata.language (ví dụ "vi") thì nội dung phải đúng ngôn ngữ đó

    def check(self, record: dict[str, Any]) -> str | None:
        language = record_language(record)
        expected = (record.get("metadata") or {}).get("language")
        # "unknown" (gần như chỉ có công thức, số hoặc code) không mâu thuẫn với ngôn ngữ của nguồn; tiếng Việt có dấu luôn được giữ (ưu tiên tiếng Việt).
        if self.match_source_language and expected and language not in (expected, "unknown", "vi"): return f"nguồn ghi ngôn ngữ {expected} nhưng nội dung là {language}"
        if language not in self.allowed: return f"ngôn ngữ {language} không nằm trong danh sách giữ ({', '.join(self.allowed)})"
        return None


@dataclass(frozen=True)
class LengthFilter:
    enabled: bool = True
    min_prompt_chars: int = 3
    min_answer_chars: int = 2
    max_total_chars: int = 16000  # khoảng 4700 token; dài hơn thì khi train bị cắt mất phần lớn câu trả lời

    def check(self, record: dict[str, Any]) -> str | None:
        prompt, answer = (text.strip() for text in conversation(record))
        if len(prompt) < self.min_prompt_chars: return f"câu hỏi quá ngắn ({len(prompt)} ký tự, tối thiểu {self.min_prompt_chars})"
        if len(answer) < self.min_answer_chars: return f"câu trả lời quá ngắn ({len(answer)} ký tự, tối thiểu {self.min_answer_chars})"
        if len(prompt) + len(answer) > self.max_total_chars: return f"quá dài ({len(prompt) + len(answer)} ký tự, tối đa {self.max_total_chars})"
        return None


def longest_word_run(words: list[str]) -> int:
    """Số lần dài nhất một từ lặp liền nhau. Chỉ tính từ có ít nhất 2 chữ cái: số và ký hiệu một chữ (ma trận, bảng) lặp là bình thường."""
    best = run = 0
    for index, word in enumerate(words):
        run = run + 1 if index and word == words[index - 1] else 1
        if sum(char.isalpha() for char in word) >= 2: best = max(best, run)
    return best


def duplicate_sentence_ratio(text: str, min_chars: int = 20) -> float:
    """Tỉ lệ câu (từ min_chars ký tự trở lên) là bản lặp của một câu đứng trước."""
    sentences = [normalize_text(part) for part in SENTENCE.split(text)]
    sentences = [sentence for sentence in sentences if len(sentence) >= min_chars]
    return 1 - len(set(sentences)) / len(sentences) if sentences else 0.0


def duplicate_ngram_ratio(words: list[str], n: int) -> float:
    """Tỉ lệ vị trí có cụm n từ đã xuất hiện trước đó (văn bản lặp vòng thì tỉ lệ này gần 1)."""
    grams = [tuple(words[index:index + n]) for index in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@dataclass(frozen=True)
class RepetitionFilter:
    enabled: bool = True
    min_words: int = 50  # văn bản (đã bỏ code) ngắn hơn thì không xét tỉ lệ lặp
    max_word_run: int = 8  # một từ lặp liền nhau quá số lần này ("ha ha ha ...") thì loại
    max_duplicate_sentence_ratio: float = 0.5  # ngưỡng cao: lời giải toán, giải thích code hay nhắc lại vài câu là bình thường
    ngram_words: int = 5
    max_duplicate_ngram_ratio: float = 0.5

    def check(self, record: dict[str, Any]) -> str | None:
        """Xét từng lượt riêng: hội thoại nhiều lượt nhắc lại ý của lượt trước là bình thường, một lượt tự lặp vòng mới là lỗi."""
        for turn, message in enumerate(sft_messages(record), start=1):
            label = f"lượt {turn} ({message['role']})"
            text = prose(str(message["content"])); words = WORD.findall(normalize_text(text))
            run = longest_word_run(words)
            if run > self.max_word_run: return f"{label} có một từ lặp liền {run} lần"
            if len(words) < self.min_words: continue
            ratio = duplicate_sentence_ratio(text)
            if ratio > self.max_duplicate_sentence_ratio: return f"{label} lặp câu ({ratio:.0%} số câu là bản lặp)"
            ratio = duplicate_ngram_ratio(words, self.ngram_words)
            if ratio > self.max_duplicate_ngram_ratio: return f"{label} lặp cụm {self.ngram_words} từ ({ratio:.0%})"
        return None


def shingles(text: str, size: int) -> set[str]:
    """Các cụm `size` từ liên tiếp của văn bản đã chuẩn hóa; văn bản ngắn hơn thì lấy cả văn bản."""
    words = WORD.findall(normalize_text(text))
    return {" ".join(words[index:index + size]) for index in range(max(1, len(words) - size + 1))} if words else set()


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


class MinHash:
    """MinHash tự viết, kiểu một hoán vị (one permutation hashing, có bước lấp ngăn rỗng bằng cách xoay vòng).

    Mỗi cụm từ chỉ băm một lần (blake2b 64 bit, có muối `seed`); mã băm chia vào `size` ngăn theo phần dư, mỗi ngăn giữ
    giá trị nhỏ nhất. Ngăn rỗng lấy giá trị của ngăn có dữ liệu gần nhất bên phải, cộng thêm khoảng cách × OFFSET để không
    trùng nhầm. Xác suất hai chữ ký bằng nhau ở một vị trí xấp xỉ độ tương đồng Jaccard của hai tập cụm từ, như MinHash
    thường với `size` hoán vị, nhưng nhanh hơn nhiều khi viết bằng Python thuần.
    """

    OFFSET = 1 << 64

    def __init__(self, size: int, seed: int):
        self.size, self.salt = size, seed.to_bytes(8, "big")

    def signature(self, items: set[str]) -> tuple[int, ...]:
        bins: list[int | None] = [None] * self.size
        for item in items:
            value = int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=8, salt=self.salt).digest(), "big")
            rest, slot = divmod(value, self.size)
            if bins[slot] is None or rest < bins[slot]: bins[slot] = rest
        if all(value is None for value in bins): return tuple([0] * self.size)
        result = []
        for index in range(self.size):
            distance = 0
            while bins[(index + distance) % self.size] is None: distance += 1
            result.append(bins[(index + distance) % self.size] + distance * self.OFFSET)
        return tuple(result)


def estimated_jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return sum(a == b for a, b in zip(left, right)) / len(left)


@dataclass(frozen=True)
class NearDuplicateFilter:
    enabled: bool = True
    threshold: float = 0.8  # độ tương đồng Jaccard (trên tập cụm từ) từ mức này trở lên thì coi là gần trùng
    shingle_words: int = 3
    permutations: int = 128  # độ dài chữ ký MinHash
    bands: int = 32  # LSH: chia chữ ký thành các dải; hai bản ghi trùng hết một dải thì mới so kỹ bằng Jaccard thật
    seed: int = 17

    def find(self, records: list[dict[str, Any]]) -> dict[int, tuple[int, float]]:
        """Vị trí bản ghi gần trùng -> (vị trí bản ghi được giữ, Jaccard). Giữ bản ghi đứng trước; trong pipeline, bản ghi đã được
        sắp theo id ở bước loại trùng tuyệt đối, nên kết quả không phụ thuộc thứ tự đọc."""
        if self.permutations % self.bands: raise ValueError("near_duplicate.permutations phải chia hết cho near_duplicate.bands")
        rows = self.permutations // self.bands
        minhash = MinHash(self.permutations, self.seed)
        sets = [shingles("\n".join(conversation(record)), self.shingle_words) for record in records]
        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        duplicates: dict[int, tuple[int, float]] = {}
        for index, items in enumerate(sets):
            signature = minhash.signature(items)
            keys = [(band, signature[band * rows:(band + 1) * rows]) for band in range(self.bands)]
            candidates = sorted({other for key in keys for other in buckets.get(key, []) if other not in duplicates})
            for other in candidates:
                similarity = jaccard(items, sets[other])  # so kỹ bằng Jaccard thật, không dựa vào ước lượng
                if similarity >= self.threshold:
                    duplicates[index] = (other, round(similarity, 3)); break
            if index not in duplicates:
                for key in keys: buckets.setdefault(key, []).append(index)
        return duplicates


@dataclass(frozen=True)
class QualityConfig:
    language: LanguageFilter = field(default_factory=LanguageFilter)
    length: LengthFilter = field(default_factory=LengthFilter)
    repetition: RepetitionFilter = field(default_factory=RepetitionFilter)
    near_duplicate: NearDuplicateFilter = field(default_factory=NearDuplicateFilter)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QualityConfig":
        groups = {item.name: item.default_factory for item in fields(cls)}
        unknown = set(value) - set(groups) - {"_comment"}
        if unknown: raise ValueError(f"Cấu hình chất lượng có nhóm lạ: {', '.join(sorted(unknown))}; chỉ có {', '.join(groups)}")
        parts = {}
        for name, factory in groups.items():
            options = {key: tuple(option) if isinstance(option, list) else option for key, option in (value.get(name) or {}).items()}
            allowed = {item.name for item in fields(factory)}
            if set(options) - allowed: raise ValueError(f"Nhóm {name} có khóa lạ: {', '.join(sorted(set(options) - allowed))}")
            parts[name] = factory(**options)
        config = cls(**parts)
        if config.near_duplicate.permutations % config.near_duplicate.bands: raise ValueError("near_duplicate.permutations phải chia hết cho near_duplicate.bands")
        if not 0 < config.near_duplicate.threshold <= 1: raise ValueError("near_duplicate.threshold phải nằm trong khoảng (0, 1]")
        return config

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_QUALITY) -> "QualityConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def quality_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Số dòng, ngôn ngữ phát hiện được, tỉ lệ tiếng Việt và độ dài (ký tự của hội thoại) của một tập bản ghi."""
    languages = Counter(record_language(record) for record in records)
    lengths = sorted(len(prompt) + len(answer) for prompt, answer in map(conversation, records))
    return {"examples": len(records), "languages": dict(sorted(languages.items())), "vietnamese_share": round(languages["vi"] / len(records), 4) if records else 0.0,
            "characters": {"mean": round(sum(lengths) / len(lengths), 1) if lengths else 0, "median": lengths[len(lengths) // 2] if lengths else 0, "max": lengths[-1] if lengths else 0}}


def apply_quality_filters(records: list[dict[str, Any]], config: QualityConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Chạy lần lượt: ngôn ngữ → độ dài → lặp → gần trùng. Trả về (bản ghi giữ lại, bản ghi bị loại kèm lý do, thống kê cho manifest)."""
    kept, rejected = [], []
    removed = {name: 0 for name in ("language", "length", "repetition", "near_duplicate")}
    for record in records:
        for name in ("language", "length", "repetition"):
            active = getattr(config, name)
            reason = active.check(record) if active.enabled else None
            if reason:
                removed[name] += 1
                rejected.append({**record, "validation_status": "rejected", "verification": {**record.get("verification", {}), "quality_filter": name, "quality_reason": reason}})
                break
        else: kept.append(record)
    if config.near_duplicate.enabled:
        duplicates = config.near_duplicate.find(kept)
        for index, (original, similarity) in sorted(duplicates.items()):
            record = kept[index]; removed["near_duplicate"] += 1
            rejected.append({**record, "validation_status": "rejected", "verification": {**record.get("verification", {}), "quality_filter": "near_duplicate", "quality_reason": f"gần trùng với {kept[original]['id']} (Jaccard {similarity})", "near_duplicate_of": kept[original]["id"]}})
        kept = [record for index, record in enumerate(kept) if index not in duplicates]
    report = {"config": asdict(config), "before": quality_statistics(records), "after": quality_statistics(kept), "removed": removed}
    return kept, rejected, report
