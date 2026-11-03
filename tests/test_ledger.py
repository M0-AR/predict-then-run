from __future__ import annotations

import yaml

from ptr.ledger import freeze, verify

PRED = [
    {
        "id": "T1",
        "experiment": "toy",
        "claim": "a toy claim",
        "kind": "threshold",
        "comparator": ">",
        "threshold": 0.0,
        "predicted_truth": True,
        "predicted_interval": [0.0, 1.0],
        "confidence": 0.7,
        "falsifier": "ci below zero",
        "citation": "none",
    }
]


def _write(tmp_path, preds):
    path = tmp_path / "predictions.yaml"
    path.write_text(yaml.safe_dump(preds, sort_keys=False))
    return path


def test_freeze_and_verify(tmp_path):
    path = _write(tmp_path, PRED)
    ledger = tmp_path / "ledger.jsonl"
    freeze(path, ledger)
    check = verify(path, ledger)
    assert check["valid"]
    assert check["entries"] == 1


def test_verify_detects_payload_change(tmp_path):
    path = _write(tmp_path, PRED)
    ledger = tmp_path / "ledger.jsonl"
    freeze(path, ledger)
    changed = [dict(PRED[0])]
    changed[0]["claim"] = "a different claim"
    path.write_text(yaml.safe_dump(changed, sort_keys=False))
    check = verify(path, ledger)
    assert not check["valid"]
    assert any("payload changed" in e for e in check["errors"])


def test_verify_detects_chain_change(tmp_path):
    path = _write(tmp_path, PRED)
    ledger = tmp_path / "ledger.jsonl"
    freeze(path, ledger)
    lines = ledger.read_text().splitlines()
    lines[0] = lines[0].replace('"seq": 0', '"seq": 99')
    ledger.write_text("\n".join(lines) + "\n")
    check = verify(path, ledger)
    assert not check["valid"]
    assert any("hash mismatch" in e for e in check["errors"])
