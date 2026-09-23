from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET_TOKENS = 10_000_000_000_000
DOMAINS = {"general_web_knowledge": .35, "long_form_technical": .10, "code": .20, "science_math": .10, "reasoning": .05, "software_engineering": .05, "agent_tool_use": .03, "quant_finance_trading": .05, "multilingual": .05, "verified_synthetic": .02}
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|(?:api[_-]?key|password|secret)\s*[=:]\s*[^\s]{8,})", re.I)


def utcnow() -> str: return datetime.now(timezone.utc).isoformat()
def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


@dataclass(frozen=True)
class Source:
    source_id: str; dataset_name: str; dataset_version: str; url: str; license: str; license_url: str; domain: str; language: str; estimated_size: int; estimated_tokens: int; download_method: str; processing_recipe: str
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        missing = [key for key in cls.__dataclass_fields__ if data.get(key) in (None, "")]
        if missing: raise ValueError(f"Source manifest missing required fields: {', '.join(missing)}")
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})


class Registry:
    def __init__(self, storage: Path):
        storage.mkdir(parents=True, exist_ok=True); self.path = storage / "registry.sqlite"; self.db = sqlite3.connect(self.path)
        self.db.executescript("""CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY, manifest TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(document_id TEXT PRIMARY KEY, source_id TEXT, domain TEXT, token_count INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS shards(shard_id TEXT PRIMARY KEY, dataset_version TEXT, path TEXT, checksum TEXT, token_count INTEGER, document_count INTEGER, domain TEXT, status TEXT, hf_path TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS processing_jobs(job_id TEXT PRIMARY KEY, source_id TEXT, status TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS uploads(shard_id TEXT PRIMARY KEY, status TEXT, hf_path TEXT, attempts INTEGER, updated_at TEXT);
CREATE TABLE IF NOT EXISTS tokenizer_versions(name TEXT PRIMARY KEY, version TEXT, checksum TEXT);
CREATE TABLE IF NOT EXISTS dataset_versions(version TEXT PRIMARY KEY, target_tokens INTEGER, status TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS experiments(experiment_id TEXT PRIMARY KEY, detail TEXT, created_at TEXT);""")
        self.db.commit()
    def source(self, source: Source) -> None: self.db.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?)", (source.source_id, json.dumps(source.__dict__, sort_keys=True), "eligible")); self.db.commit()
    def shard(self, values: tuple[Any, ...]) -> None: self.db.execute("INSERT OR REPLACE INTO shards VALUES(?,?,?,?,?,?,?,?,?,?)", values); self.db.commit()
    def status(self, shard_id: str, status: str, hf_path: str | None = None) -> None:
        self.db.execute("UPDATE shards SET status=? WHERE shard_id=?", (status, shard_id))
        self.db.execute("INSERT INTO uploads(shard_id,status,hf_path,attempts,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(shard_id) DO UPDATE SET status=excluded.status, hf_path=excluded.hf_path, attempts=uploads.attempts+1, updated_at=excluded.updated_at", (shard_id, status, hf_path or "", 1, utcnow()))
        self.db.commit()
    def existing(self, shard_id: str) -> str | None:
        row = self.db.execute("SELECT status FROM shards WHERE shard_id=?", (shard_id,)).fetchone(); return row[0] if row else None
    def progress(self, target: int) -> dict[str, Any]:
        total = lambda statuses: self.db.execute(f"SELECT COALESCE(SUM(token_count),0) FROM shards WHERE status IN ({','.join('?' for _ in statuses)})", statuses).fetchone()[0]
        acquired = total(("PLANNED", "VALIDATED", "UPLOADING", "REMOTE_VERIFIED", "COMPLETE", "FAILED"))
        validated = total(("VALIDATED", "UPLOADING", "REMOTE_VERIFIED", "COMPLETE"))
        uploaded = total(("UPLOADING", "REMOTE_VERIFIED", "COMPLETE"))
        verified = total(("REMOTE_VERIFIED", "COMPLETE"))
        domains = dict(self.db.execute("SELECT domain, SUM(token_count) FROM shards WHERE status='COMPLETE' GROUP BY domain").fetchall())
        failed = self.db.execute("SELECT COUNT(*) FROM shards WHERE status='FAILED'").fetchone()[0]
        pending = self.db.execute("SELECT COUNT(*) FROM shards WHERE status NOT IN ('COMPLETE','FAILED')").fetchone()[0]
        return {"target_tokens": target, "acquired_tokens": acquired, "validated_tokens": validated, "approved_tokens": validated, "uploaded_tokens": uploaded, "remotely_verified_tokens": verified, "rejected_tokens": 0, "remaining_tokens": max(target - verified, 0), "percentage_complete": verified / target * 100, "by_domain": domains, "failed_shards": failed, "pending_shards": pending}


class Tokenizer:
    """Token accounting is always derived from this configured tokenizer, never character estimates."""
    def __init__(self, config: dict[str, Any]):
        self.name = config["name"]; self.kind = config["kind"]; self.revision = config.get("revision", "unspecified")
        if self.kind == "huggingface":
            try:
                from transformers import AutoTokenizer
            except ImportError as error: raise RuntimeError("Install transformers for the configured production tokenizer") from error
            self.encoder = AutoTokenizer.from_pretrained(config["source"], revision=config.get("revision", "main"), local_files_only=config.get("offline", False))
        elif self.kind == "tiktoken":
            try:
                import tiktoken
            except ImportError as error: raise RuntimeError("Install tiktoken for this configured tokenizer") from error
            self.encoder = tiktoken.get_encoding(config["encoding"])
        elif self.kind != "utf8_bytes": raise ValueError("Supported tokenizer kinds: huggingface, tiktoken, utf8_bytes")
    def encode(self, text: str) -> list[int]:
        if self.kind == "huggingface": return self.encoder.encode(text, add_special_tokens=False)
        return self.encoder.encode(text) if self.kind == "tiktoken" else list(text.encode("utf-8"))
    def info(self) -> dict[str, str]:
        details = {"name": self.name, "kind": self.kind, "revision": self.revision}
        return {**details, "hash": hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()}


class HuggingFaceUploader:
    def __init__(self, repo: str | None, enabled: bool, retries: int = 3):
        self.enabled, self.repo, self.retries = enabled, repo, retries
        if enabled and not os.environ.get("HF_TOKEN"): raise RuntimeError("HF_TOKEN is required when Hugging Face upload is enabled")
        if enabled and not repo: raise RuntimeError("HF_DATASET_REPO or upload.repo is required when upload is enabled")
    def upload_verify(self, local: Path, remote: str, checksum: str) -> str:
        if not self.enabled: return "LOCAL_VALIDATED"
        try:
            from huggingface_hub import HfApi
        except ImportError as error: raise RuntimeError("Install huggingface_hub to enable uploads") from error
        api = HfApi(token=os.environ["HF_TOKEN"])
        for attempt in range(self.retries):
            try:
                if not api.file_exists(self.repo, remote, repo_type="dataset"):
                    api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=self.repo, repo_type="dataset", commit_message=f"Upload shard {local.stem}")
                if api.file_exists(self.repo, remote, repo_type="dataset"): return "REMOTE_VERIFIED"
            except Exception:
                if attempt + 1 == self.retries: raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Remote verification failed")


def storage_paths(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ("raw", "cache", "processed", "tokenized", "shards", "rejected", "manifests", "logs", "checkpoints")}
    for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
    return paths


def scan_secrets(root: Path) -> list[str]:
    findings = []
    for path in root.rglob("*"):
        if path.is_file() and path.name != "corpus.py" and ".git" not in path.parts and path.stat().st_size < 2_000_000:
            try:
                if SECRET.search(path.read_text(encoding="utf-8", errors="ignore")): findings.append(str(path))
            except OSError: pass
    return findings


def acquire(source: Source, paths: dict[str, Path], registry: Registry, dry_run: bool) -> Path:
    registry.source(source); target = paths["raw"] / f"{source.source_id}.txt"
    if target.exists(): return target
    if dry_run: return target
    if source.download_method != "http": raise ValueError(f"Unsupported download method: {source.download_method}")
    with urllib.request.urlopen(source.url, timeout=60) as response, target.open("wb") as output: shutil.copyfileobj(response, output)
    return target


def clean_text(text: str) -> tuple[str | None, str | None]:
    if SECRET.search(text): return None, "pii_or_secret"
    text = text.strip()
    if len(text) < 80: return None, "too_short"
    if len(set(text)) < 10: return None, "low_quality"
    return text, None


def build_one(source: Source, config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    root = Path(config["storage_root"]); paths = storage_paths(root); registry = Registry(root)
    tokenizer = Tokenizer(config["tokenizer"]); registry.db.execute("INSERT OR REPLACE INTO tokenizer_versions VALUES(?,?,?)", (tokenizer.name, tokenizer.kind, tokenizer.info()["hash"])); registry.db.execute("INSERT OR REPLACE INTO dataset_versions VALUES(?,?,?,?)", (config["dataset_version"], config["target_tokens"], "building", utcnow())); registry.db.commit()
    raw = acquire(source, paths, registry, dry_run)
    if dry_run: return {"action": "would download/process/tokenize/upload", "source": source.source_id, "raw": str(raw)}
    text, reason = clean_text(raw.read_text(encoding="utf-8", errors="ignore"))
    if not text: (paths["rejected"] / f"{source.source_id}.reason").write_text(reason or "rejected"); return {"rejected": reason}
    tokens = tokenizer.encode(text); raw_bytes = raw.stat().st_size; shard_id = hashlib.sha256((source.source_id + digest(raw)).encode()).hexdigest()[:20]; remote = f"train/{source.domain}/{shard_id}.jsonl"
    if registry.existing(shard_id) == "COMPLETE": return {"shard_id": shard_id, "status": "COMPLETE", "idempotent": True}
    payload = {"shard_id": shard_id, "dataset_version": config["dataset_version"], "source": source.source_id, "source_url": source.url, "source_version": source.dataset_version, "acquired_at": utcnow(), "domain": source.domain, "language": source.language, "raw_byte_count": raw_bytes, "accepted_byte_count": len(text.encode("utf-8")), "tokenizer": tokenizer.info(), "token_count": len(tokens), "document_count": 1, "created_at": utcnow(), "processing_version": "corpus-v1", "license_summary": source.license, "validation_status": "VALIDATED", "upload_status": "PENDING", "tokens": tokens}
    local = paths["shards"] / f"{shard_id}.jsonl"; local.write_text(json.dumps(payload, separators=(",", ":")) + "\n"); checksum = digest(local); payload["checksum"] = checksum; local.write_text(json.dumps(payload, separators=(",", ":")) + "\n"); checksum = digest(local)
    uploader = HuggingFaceUploader(config.get("upload", {}).get("repo") or os.getenv("HF_DATASET_REPO"), config.get("upload", {}).get("enabled", False), config.get("upload", {}).get("retries", 3))
    registry.shard((shard_id, config["dataset_version"], str(local), checksum, len(tokens), 1, source.domain, "VALIDATED", remote, utcnow()))
    registry.status(shard_id, "UPLOADING", remote) if uploader.enabled else None
    try:
        status = uploader.upload_verify(local, remote, checksum)
    except Exception:
        registry.status(shard_id, "FAILED", remote); raise
    complete = "COMPLETE" if status == "REMOTE_VERIFIED" else "VALIDATED"
    if status == "REMOTE_VERIFIED": registry.status(shard_id, "REMOTE_VERIFIED", remote)
    registry.status(shard_id, complete, remote)
    if complete == "COMPLETE" and config.get("retention", {}).get("raw_retention") == "delete": raw.unlink(missing_ok=True)
    manifest = {**payload, "checksum": checksum, "byte_size": local.stat().st_size, "upload_status": status, "hf_path": remote}; (paths["manifests"] / f"{shard_id}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {key: value for key, value in manifest.items() if key != "tokens"}
