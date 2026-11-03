from __future__ import annotations

import numpy as np

from ptr.predictions import aggregate_scores, calibration_bins, score_prediction


def _pred(kind="directional", **kw):
    base = {
        "id": "P", "experiment": "toy", "claim": "claim", "kind": kind,
        "confidence": 0.8, "falsifier": "f", "citation": "c",
        "predicted_relative_effect": 0.1, "predicted_interval": [0.0, 0.2],
    }
    if kind == "threshold":
        base.update({"comparator": ">", "threshold": 0.0, "predicted_truth": True,
                     "predicted_interval": [0.0, 0.2]})
    base.update(kw)
    return base


def test_directional_confirmed():
    out = score_prediction(_pred(), {"measured_relative": 0.15, "ci_low_relative": 0.05, "ci_high_relative": 0.25})
    assert out["verdict"] == "CONFIRMED"
    assert out["outcome"] == 1


def test_directional_refuted():
    out = score_prediction(_pred(), {"measured_relative": -0.15, "ci_low_relative": -0.25, "ci_high_relative": -0.05})
    assert out["verdict"] == "REFUTED"
    assert out["outcome"] == 0


def test_directional_inconclusive():
    out = score_prediction(_pred(), {"measured_relative": 0.05, "ci_low_relative": -0.01, "ci_high_relative": 0.1})
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["outcome"] is None


def test_threshold_confirmed_and_refuted():
    out = score_prediction(_pred("threshold"), {"measured": 0.2, "ci_low": 0.1, "ci_high": 0.3})
    assert out["verdict"] == "CONFIRMED"
    out = score_prediction(_pred("threshold"), {"measured": -0.2, "ci_low": -0.3, "ci_high": -0.1})
    assert out["verdict"] == "REFUTED"


def test_aggregate_and_bins():
    scores = [
        score_prediction(_pred(), {"measured_relative": 0.15, "ci_low_relative": 0.05, "ci_high_relative": 0.25}),
        score_prediction(_pred(), {"measured_relative": -0.15, "ci_low_relative": -0.25, "ci_high_relative": -0.05}),
    ]
    agg = aggregate_scores(scores)
    assert agg["n_predictions"] == 2
    assert agg["n_confirmed"] == 1
    assert agg["n_refuted"] == 1
    assert np.isfinite(agg["brier"])
    bins = calibration_bins(scores)
    assert sum(b["n"] for b in bins) == 2
