"""Run records and verified stage caching."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from .evidence import require, sha256


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_object(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def versions() -> dict:
    result = {}

    for package in [
        "openai",
        "openai-agents",
        "mcp",
        "catboost",
        "pandas",
        "numpy",
        "pydantic",
        "Pillow",
    ]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"

    return result


class Audit:
    def __init__(self, output: Path, fingerprint: dict):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "run_manifest.json"

        if self.path.exists():
            self.state = json.loads(
                self.path.read_text(encoding="utf-8")
            )

            require(
                self.state["input_hash"] == digest_object(fingerprint),
                "Inputs, code, skill, dependencies, or models changed. "
                "Use a new output directory.",
            )
        else:
            require(
                not any(self.output.iterdir()),
                "Use an empty output directory for a new run.",
            )

            self.state = {
                "status": "not_started",
                "created_at": now(),
                "input_hash": digest_object(fingerprint),
                "inputs": fingerprint,
                "stages": {},
                "versions": versions(),
            }

            self.save()

    def save(self):
        self.state["updated_at"] = now()
        write_json(self.path, self.state)

    def event(self, kind: str, **details):
        event = {
            "timestamp": now(),
            "kind": kind,
            **details,
        }

        with (self.output / "trace.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )

    def cached(self, stage: str):
        saved = self.state["stages"].get(stage)

        if not saved:
            return None

        path = self.output / saved["file"]

        require(
            path.is_file()
            and sha256(path) == saved["sha256"],
            f"Cached {stage} output is missing or changed; "
            "do not silently reuse it.",
        )

        self.event(
            "stage_cache_hit",
            stage=stage,
            output_sha256=saved["sha256"],
        )

        return json.loads(path.read_text(encoding="utf-8"))

    def complete_stage(self, stage: str, value):
        path = self.output / f"{stage}.json"
        write_json(path, value)
        output_hash = sha256(path)

        self.state["stages"][stage] = {
            "file": path.name,
            "sha256": output_hash,
            "completed_at": now(),
        }

        self.save()

        self.event(
            "stage_completed",
            stage=stage,
            output_sha256=output_hash,
        )

    def fail(self, error: Exception):
        message = str(error)
        key = os.environ.get("OPENAI_API_KEY")

        if key:
            message = message.replace(key, "[REDACTED]")

        self.state["status"] = "failed"
        self.state["error"] = {
            "type": type(error).__name__,
            "message": message[:2000],
        }

        self.save()
        self.event("run_failed", **self.state["error"])
