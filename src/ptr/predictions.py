from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .stats import brier, brier_skill_score, log_loss


def load_predictions(path: str | Path = "preregistry/predictions.yaml") -> list[dict]:
    preds = yaml.safe_load(Path(path).read_text())
    ids = [p["id"] for p in preds]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prediction ids")
    for p in preds:
        for key in ("id", "experiment", "claim", "kind", "confidence", "falsifier", "citation"):
            if key not in p:
                raise ValueError(f"prediction {p.get('id')} missing {key}")
    return preds


def _rate(measured: float, ci_low: float, ci_high: float, comparator: str, threshold: float) -> bool | None:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return None
    if comparator in ("<", "<="):
        if ci_high < threshold:
            return True
        if ci_low > threshold:
            return False
    elif comparator in (">", ">="):
        if ci_low > threshold:
            return True
        if ci_high < threshold:
            return False
    else:
        raise ValueError(f"unknown comparator {comparator}")
    return None


def score_prediction(pred: dict, result: dict) -> dict:
    kind = pred["kind"]
    if kind == "directional":
        measured = float(result["measured_relative"])
        lo = float(result["ci_low_relative"])
        hi = float(result["ci_high_relative"])
        sign_ok = np.sign(measured) == np.sign(pred["predicted_relative_effect"])
        excludes_zero = lo > 0 or hi < 0
        verdict = "CONFIRMED" if (excludes_zero and sign_ok) else ("REFUTED" if excludes_zero else "INCONCLUSIVE")
        covered = bool(pred["predicted_interval"][0] <= measured <= pred["predicted_interval"][1])
        target = float(pred["predicted_relative_effect"])
        abs_err = abs(measured - target)
        rel_err = abs_err / max(abs(measured), abs(target), 1e-9)
        outcome = 1 if verdict == "CONFIRMED" else (0 if verdict == "REFUTED" else None)
    elif kind == "threshold":
        measured = float(result["measured"])
        lo = float(result["ci_low"])
        hi = float(result["ci_high"])
        truth = _rate(measured, lo, hi, pred["comparator"], float(pred["threshold"]))
        if truth is None:
            verdict = "INCONCLUSIVE"
            outcome = None
        else:
            confirmed = truth == bool(pred["predicted_truth"])
            verdict = "CONFIRMED" if confirmed else "REFUTED"
            outcome = 1 if confirmed else 0
        covered = bool(pred["predicted_interval"][0] <= measured <= pred["predicted_interval"][1])
        target = float(pred["threshold"])
        abs_err = abs(measured - target)
        rel_err = abs_err / max(abs(measured), abs(target), 1e-9)
    else:
        raise ValueError(f"unknown prediction kind {kind}")
    return {
        "id": pred["id"],
        "experiment": pred["experiment"],
        "claim": pred["claim"],
        "kind": kind,
        "confidence": float(pred["confidence"]),
        "measured": measured,
        "ci_low": lo,
        "ci_high": hi,
        "verdict": verdict,
        "outcome": outcome,
        "sign_match": bool(sign_ok) if kind == "directional" else None,
        "interval_covered": covered,
        "absolute_error": abs_err,
        "relative_error": rel_err,
        "result": result,
    }


def aggregate_scores(scores: list[dict]) -> dict:
    decided = [s for s in scores if s["outcome"] is not None]
    p = [s["confidence"] for s in decided]
    y = [s["outcome"] for s in decided]
    out = {
        "n_predictions": len(scores),
        "n_decided": len(decided),
        "n_confirmed": sum(1 for s in scores if s["verdict"] == "CONFIRMED"),
        "n_refuted": sum(1 for s in scores if s["verdict"] == "REFUTED"),
        "n_inconclusive": sum(1 for s in scores if s["verdict"] == "INCONCLUSIVE"),
        "brier": brier(p, y) if decided else float("nan"),
        "brier_skill_vs_half": brier_skill_score(p, y) if decided else float("nan"),
        "log_loss": log_loss(p, y) if decided else float("nan"),
        "interval_coverage": float(np.mean([s["interval_covered"] for s in scores])) if scores else float("nan"),
        "mean_absolute_error": float(np.mean([s["absolute_error"] for s in scores])) if scores else float("nan"),
        "mean_relative_error": float(np.mean([s["relative_error"] for s in scores])) if scores else float("nan"),
        "sign_accuracy": float(np.mean([s["sign_match"] for s in decided if s["sign_match"] is not None]))
        if any(s["sign_match"] is not None for s in decided) else float("nan"),
    }
    return out


def calibration_bins(scores: list[dict], n_bins: int = 5) -> list[dict]:
    decided = [s for s in scores if s["outcome"] is not None]
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        sel = [s for s in decided if edges[i] <= s["confidence"] < edges[i + 1] or (i == n_bins - 1 and s["confidence"] == 1.0)]
        if sel:
            bins.append({
                "lo": float(edges[i]), "hi": float(edges[i + 1]),
                "n": len(sel), "mean_confidence": float(np.mean([s["confidence"] for s in sel])),
                "empirical_accuracy": float(np.mean([s["outcome"] for s in sel])),
            })
    return bins
