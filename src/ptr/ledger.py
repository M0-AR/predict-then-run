from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except Exception:
        return "uncommitted", True


def freeze(predictions_path: str | Path, ledger_path: str | Path) -> list[dict]:
    preds = yaml.safe_load(Path(predictions_path).read_text())
    entries = []
    prev = "GENESIS"
    commit, dirty = _git_state()
    for i, pred in enumerate(preds):
        payload = canonical_json(pred)
        payload_hash = sha256_text(payload)
        entry = {
            "seq": i,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prediction_id": pred["id"],
            "payload_sha256": payload_hash,
            "prev_hash": prev,
            "git_commit": commit,
            "git_dirty": dirty,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        entry["entry_hash"] = sha256_text(canonical_json(entry))
        entries.append(entry)
        prev = entry["entry_hash"]
    out = Path(ledger_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(e, sort_keys=True) for e in entries) + "\n")
    return entries


def verify(predictions_path: str | Path, ledger_path: str | Path) -> dict:
    preds = yaml.safe_load(Path(predictions_path).read_text())
    by_id = {p["id"]: p for p in preds}
    lines = [json.loads(line) for line in Path(ledger_path).read_text().splitlines() if line.strip()]
    prev = "GENESIS"
    errors = []
    for entry in lines:
        body = {k: v for k, v in entry.items() if k != "entry_hash"}
        if sha256_text(canonical_json(body)) != entry["entry_hash"]:
            errors.append(f"entry {entry['seq']} hash mismatch")
        if entry["prev_hash"] != prev:
            errors.append(f"entry {entry['seq']} chain mismatch")
        pred = by_id.get(entry["prediction_id"])
        if pred is None:
            errors.append(f"missing prediction {entry['prediction_id']}")
        elif sha256_text(canonical_json(pred)) != entry["payload_sha256"]:
            errors.append(f"payload changed after freeze: {entry['prediction_id']}")
        prev = entry["entry_hash"]
    return {
        "valid": not errors,
        "entries": len(lines),
        "errors": errors,
        "head": prev,
    }
