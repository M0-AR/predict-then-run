from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats


def bootstrap_ci(values: Iterable[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(a.mean()), float(lo), float(hi)


def paired_diff_ci(a: Iterable[float], b: Iterable[float], n_boot: int = 10000,
                   alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    d = a - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def welch_p(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def holm(pvalues: Iterable[float]) -> list[float]:
    p = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted.tolist()


def brier(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    bs = np.mean((p - y) ** 2)
    ref = np.mean((0.5 - y) ** 2)
    return float(1 - bs / ref) if ref > 0 else float("nan")


def log_loss(probabilities: Iterable[float], outcomes: Iterable[int], eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(list(probabilities), dtype=float), eps, 1 - eps)
    y = np.asarray(list(outcomes), dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sign_accuracy(predicted: Iterable[float], measured: Iterable[float], tol: float = 0.0) -> float:
    p = np.asarray(list(predicted), dtype=float)
    m = np.asarray(list(measured), dtype=float)
    return float(np.mean(np.sign(p) == np.sign(m))) if p.size else float("nan")


def interval_coverage(measured: Iterable[float], lows: Iterable[float],
                      highs: Iterable[float]) -> float:
    m = np.asarray(list(measured), dtype=float)
    lo = np.asarray(list(lows), dtype=float)
    hi = np.asarray(list(highs), dtype=float)
    return float(np.mean((m >= lo) & (m <= hi))) if m.size else float("nan")


def relative_error(measured: float, predicted: float) -> float:
    return float(abs(measured - predicted) / (abs(measured) + 1e-12))


def fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{nd}f}"
