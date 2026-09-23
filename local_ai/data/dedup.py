"""Loại trùng gần đúng (near-dedup) bằng MinHash + LSH, lưu chỉ số trong registry SQLite để so giữa các nguồn.

Mỗi văn bản được chia thành các "tài liệu" (đoạn cách nhau bởi dòng trống). Tài liệu có độ giống
(Jaccard ước lượng trên shingle từ) >= threshold với tài liệu đã có thì bị loại, để không đếm trùng token.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Any

PRIME = (1 << 61) - 1
WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class NearDedupSettings:
    enabled: bool = True
    threshold: float = 0.8
    num_perm: int = 128
    bands: int = 32
    shingle_words: int = 5
    min_words: int = 20

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NearDedupSettings":
        settings = cls(**config.get("near_dedup", {}))
        if settings.num_perm % settings.bands: raise ValueError("near_dedup.num_perm phải chia hết cho near_dedup.bands")
        if not 0 < settings.threshold <= 1: raise ValueError("near_dedup.threshold phải nằm trong (0, 1]")
        return settings


def _permutations(count: int) -> list[tuple[int, int]]:
    # Hệ số cố định (sinh từ sha256) để chữ ký giống nhau giữa các lần chạy.
    values = []
    for index in range(count):
        seed = hashlib.sha256(f"minhash-{index}".encode()).digest()
        values.append((int.from_bytes(seed[:8], "big") % (PRIME - 1) + 1, int.from_bytes(seed[8:16], "big") % PRIME))
    return values


def shingles(text: str, size: int) -> set[int]:
    words = WORD.findall(unicodedata.normalize("NFC", text).casefold())
    grams = [" ".join(words[i:i + size]) for i in range(max(len(words) - size + 1, 1))] if words else []
    return {int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big") for gram in grams}


def signature(items: set[int], permutations: list[tuple[int, int]]) -> list[int]:
    return [min((a * item + b) % PRIME for item in items) for a, b in permutations]


def estimated_jaccard(left: list[int], right: list[int]) -> float:
    return sum(x == y for x, y in zip(left, right)) / len(left)


def split_documents(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


class NearDedupIndex:
    def __init__(self, db: sqlite3.Connection, settings: NearDedupSettings):
        self.db, self.settings = db, settings
        self.rows = settings.num_perm // settings.bands; self.permutations = _permutations(settings.num_perm)
        db.executescript("""CREATE TABLE IF NOT EXISTS minhash_signatures(document_id TEXT PRIMARY KEY, source_id TEXT, signature TEXT);
CREATE TABLE IF NOT EXISTS minhash_bands(band INTEGER, bucket TEXT, document_id TEXT);
CREATE INDEX IF NOT EXISTS minhash_band_lookup ON minhash_bands(band, bucket);""")

    def forget_source(self, source_id: str) -> None:
        """Xây lại một nguồn: xóa chỉ số cũ của nguồn đó để không tự coi là trùng với chính mình."""
        self.db.execute("DELETE FROM minhash_bands WHERE document_id IN (SELECT document_id FROM minhash_signatures WHERE source_id=?)", (source_id,))
        self.db.execute("DELETE FROM minhash_signatures WHERE source_id=?", (source_id,)); self.db.commit()

    def _buckets(self, sig: list[int]) -> list[tuple[int, str]]:
        return [(band, hashlib.sha1(json.dumps(sig[band * self.rows:(band + 1) * self.rows]).encode()).hexdigest()) for band in range(self.settings.bands)]

    def find_duplicate(self, sig: list[int]) -> str | None:
        candidates = {row[0] for band, bucket in self._buckets(sig) for row in self.db.execute("SELECT document_id FROM minhash_bands WHERE band=? AND bucket=?", (band, bucket))}
        for candidate in sorted(candidates):
            stored = self.db.execute("SELECT signature FROM minhash_signatures WHERE document_id=?", (candidate,)).fetchone()
            if stored and estimated_jaccard(sig, json.loads(stored[0])) >= self.settings.threshold: return candidate
        return None

    def add(self, document_id: str, source_id: str, sig: list[int]) -> None:
        self.db.execute("INSERT OR REPLACE INTO minhash_signatures VALUES(?,?,?)", (document_id, source_id, json.dumps(sig)))
        self.db.executemany("INSERT INTO minhash_bands VALUES(?,?,?)", [(band, bucket, document_id) for band, bucket in self._buckets(sig)])

    def filter_text(self, source_id: str, text: str) -> tuple[str, dict[str, Any]]:
        """Trả về văn bản đã bỏ tài liệu gần trùng và thống kê. Tài liệu quá ngắn (< min_words) được giữ nguyên."""
        if not self.settings.enabled: return text, {"documents": 1, "near_duplicates": 0}
        self.forget_source(source_id); kept: list[str] = []; duplicates: list[dict[str, str]] = []
        documents = split_documents(text)
        for index, document in enumerate(documents):
            if len(WORD.findall(document)) < self.settings.min_words: kept.append(document); continue
            sig = signature(shingles(document, self.settings.shingle_words), self.permutations)
            match = self.find_duplicate(sig)
            if match: duplicates.append({"document": f"{source_id}:{index}", "duplicate_of": match}); continue
            self.add(f"{source_id}:{index}", source_id, sig); kept.append(document)
        self.db.commit()
        return "\n\n".join(kept), {"documents": len(documents), "near_duplicates": len(duplicates), "examples": duplicates[:10]}
