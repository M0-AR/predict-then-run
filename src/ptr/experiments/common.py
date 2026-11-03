from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from ..config import Config
from ..data import char_dataset, load_public_text
from ..stats import bootstrap_ci, paired_diff_ci


def char_data(cfg: Config, max_chars: int | None = None):
    text, manifest = load_public_text(
        "tiny_stories",
        cache_dir=cfg.data.get("cache_dir", "data/cache"),
        max_chars=max_chars or int(cfg.data.get("max_chars", 1_200_000)),
        url=cfg.data.get("tiny_stories_url"),
    )
    ds = char_dataset(text, cfg.model.ctx)
    return ds, manifest


def summarise(values: list[float], seed: int = 0) -> dict:
    mean, lo, hi = bootstrap_ci(values, seed=seed)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": [float(v) for v in values]}


def directional_result(baseline: list[float], treatment: list[float], seed: int = 0) -> dict:
    diff, lo, hi = paired_diff_ci(treatment, baseline, seed=seed)
    base_mean = float(np.mean(baseline))
    rel = diff / (abs(base_mean) + 1e-12)
    rel_lo = lo / (abs(base_mean) + 1e-12)
    rel_hi = hi / (abs(base_mean) + 1e-12)
    p_welch = float(stats.ttest_ind(treatment, baseline, equal_var=False).pvalue) if len(baseline) > 1 and len(treatment) > 1 else float("nan")
    p_paired = float(stats.ttest_rel(treatment, baseline).pvalue) if len(baseline) > 1 and len(treatment) == len(baseline) else float("nan")
    return {
        "baseline_mean": base_mean,
        "treatment_mean": float(np.mean(treatment)),
        "measured_relative": rel,
        "ci_low_relative": rel_lo,
        "ci_high_relative": rel_hi,
        "absolute_diff": diff,
        "absolute_ci_low": lo,
        "absolute_ci_high": hi,
        "p_value": p_welch,
        "p_value_paired": p_paired,
        "n": len(baseline),
    }


def rate_result(baseline: list[float], treatment: list[float], seed: int = 0) -> dict:
    """Absolute rate difference (percentage points) with a paired bootstrap CI.

    Used for accuracy/top-1 predictions, where a relative change is undefined
    when the baseline is zero.
    """
    diff, lo, hi = paired_diff_ci(treatment, baseline, seed=seed)
    return {
        "baseline_mean": float(np.mean(baseline)),
        "treatment_mean": float(np.mean(treatment)),
        "measured_relative": diff,
        "ci_low_relative": lo,
        "ci_high_relative": hi,
        "absolute_diff": diff,
        "absolute_ci_low": lo,
        "absolute_ci_high": hi,
        "n": len(baseline),
        "unit": "rate_difference",
    }


def save_result(name: str, payload: dict, results_dir: str | Path = "results/raw") -> Path:
    out = Path(results_dir) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return out


def load_result(name: str, results_dir: str | Path = "results/raw") -> dict:
    return json.loads((Path(results_dir) / f"{name}.json").read_text())


def trend(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_stat(values: list[float], fn, n_boot: int = 5000, seed: int = 0) -> dict:
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats_ = []
    for _ in range(n_boot):
        sel = a[rng.integers(0, a.size, a.size)]
        stats_.append(fn(sel))
    return {"mean": float(np.mean(stats_)), "ci_low": float(np.quantile(stats_, 0.025)),
            "ci_high": float(np.quantile(stats_, 0.975)), "n": len(values)}
