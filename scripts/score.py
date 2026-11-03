#!/usr/bin/env python3
"""Score frozen predictions against the final raw results."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ptr.ledger import verify
from ptr.predictions import aggregate_scores, calibration_bins, load_predictions, score_prediction


def _normalise(result: dict, kind: str) -> dict:
    out = dict(result)
    if kind == "threshold" and "measured" not in out:
        out["measured"] = out.get("measured_relative")
        out["ci_low"] = out.get("ci_low_relative")
        out["ci_high"] = out.get("ci_high_relative")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="preregistry/predictions.yaml")
    ap.add_argument("--ledger", default="preregistry/ledger.jsonl")
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/scores.json")
    args = ap.parse_args()

    check = verify(args.predictions, args.ledger)
    if not check["valid"]:
        raise SystemExit("ledger invalid: " + "; ".join(check["errors"]))

    ledger_mtime = Path(args.ledger).stat().st_mtime
    raw_files = sorted(Path(args.raw).glob("*.json"))
    if not raw_files:
        raise SystemExit("no raw results found")
    latest_raw = max(p.stat().st_mtime for p in raw_files)
    if latest_raw < ledger_mtime:
        raise SystemExit("raw results predate the frozen ledger; rerun after freezing")

    results = {}
    for path in raw_files:
        payload = json.loads(path.read_text())
        for pid, res in payload.get("predictions", {}).items():
            if pid in results:
                raise SystemExit(f"duplicate result {pid} in {path}")
            results[pid] = res

    preds = load_predictions(args.predictions)
    scores = []
    for pred in preds:
        if pred["id"] not in results:
            raise SystemExit(f"missing result for {pred['id']}")
        res = _normalise(results[pred["id"]], pred["kind"])
        if pred["kind"] == "directional":
            if "measured_relative" not in res:
                raise SystemExit(f"{pred['id']} lacks measured_relative")
        scores.append(score_prediction(pred, res))

    aggregate = aggregate_scores(scores)
    bins = calibration_bins(scores)
    payload = {
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "ledger": check,
        "aggregate": aggregate,
        "calibration": bins,
        "scores": scores,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    table = Path("results/tables")
    table.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "id": s["id"],
        "experiment": s["experiment"],
        "kind": s["kind"],
        "confidence": s["confidence"],
        "measured": s["measured"],
        "ci_low": s["ci_low"],
        "ci_high": s["ci_high"],
        "verdict": s["verdict"],
        "outcome": s["outcome"],
        "interval_covered": s["interval_covered"],
        "relative_error": s["relative_error"],
        "p_value": s["result"].get("p_value"),
    } for s in scores]).to_csv(table / "scores.csv", index=False)

    print(json.dumps(aggregate, indent=2))
    print(f"wrote {args.out} and {table / 'scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
