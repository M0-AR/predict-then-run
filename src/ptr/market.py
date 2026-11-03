from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_prices(symbol: str = "SPY", range_: str = "10y", interval: str = "1d",
                 cache_dir: str | Path = "data/cache", max_age_hours: float = 24.0) -> tuple[pd.DataFrame, dict]:
    cache = Path(cache_dir) / f"{symbol}_{range_}_{interval}.csv"
    meta_path = cache.with_suffix(".json")
    fresh = False
    if cache.exists() and meta_path.exists():
        age = (time.time() - cache.stat().st_mtime) / 3600
        fresh = age < max_age_hours
    if not fresh:
        url = YAHOO.format(symbol=symbol)
        params = {"range": range_, "interval": interval}
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        last = None
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=45)
                r.raise_for_status()
                data = r.json()
                result = data["chart"]["result"][0]
                ts = result["timestamp"]
                quote = result["indicators"]["quote"][0]
                adj = result["indicators"].get("adjclose")
                close = adj[0]["adjclose"] if adj and adj[0].get("adjclose") else quote["close"]
                df = pd.DataFrame({
                    "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").date,
                    "close": close,
                }).dropna()
                df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                df.to_csv(cache, index=False)
                meta_path.write_text(json.dumps({
                    "symbol": symbol,
                    "url": url,
                    "params": params,
                    "fetched_utc": datetime.now(timezone.utc).isoformat(),
                    "rows": len(df),
                    "first": str(df["date"].iloc[0]),
                    "last": str(df["date"].iloc[-1]),
                    "price_field": "adjclose" if adj and adj[0].get("adjclose") else "close",
                }, indent=2))
                break
            except Exception as exc:  # pragma: no cover - network
                last = exc
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"market fetch failed: {last}")
    df = pd.read_csv(cache, parse_dates=["date"])
    meta = json.loads(meta_path.read_text())
    meta["sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    meta["cached"] = fresh
    return df, meta


def build_features(df: pd.DataFrame, horizon: int = 1) -> tuple[np.ndarray, np.ndarray, pd.Series, list[str]]:
    close = df["close"].astype(float).values
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    r1 = logret
    r5 = pd.Series(logret).rolling(5).sum().values
    r20 = pd.Series(logret).rolling(20).sum().values
    vol20 = pd.Series(logret).rolling(20).std().values
    sma20 = pd.Series(close).rolling(20).mean().values
    gap = close / sma20 - 1.0
    X = np.column_stack([r1, r5, r20, vol20, gap])
    y = pd.Series(logret).shift(-horizon).values
    dates = pd.to_datetime(df["date"])
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask], dates[mask].reset_index(drop=True), ["r1", "r5", "r20", "vol20", "ma_gap"]


def walk_forward(X: np.ndarray, y: np.ndarray, dates: pd.Series, min_train: int = 750,
                 test_size: int = 63, embargo: int = 1, alpha: float = 10.0,
                 cost_bps: float = 1.0, shuffle_labels: bool = False) -> dict:
    preds, actual, when = [], [], []
    fold = 0
    for start in range(min_train, len(y), test_size):
        train_end = max(0, start - embargo)
        if train_end < min_train // 2:
            continue
        Xtr, ytr = X[:train_end], y[:train_end]
        if shuffle_labels:
            rng = np.random.default_rng(9000 + fold)
            ytr = ytr[rng.permutation(len(ytr))]
        Xte, yte = X[start : start + test_size], y[start : start + test_size]
        if len(Xte) == 0:
            continue
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        A = Xtr_s.T @ Xtr_s + alpha * np.eye(Xtr_s.shape[1])
        beta = np.linalg.solve(A, Xtr_s.T @ ytr)
        preds.extend((Xte_s @ beta).tolist())
        actual.extend(yte.tolist())
        when.extend(dates.iloc[start : start + test_size].astype(str).tolist())
        fold += 1
    pred = np.asarray(preds)
    act = np.asarray(actual)
    if pred.size == 0:
        return {"n": 0, "n_folds": 0}
    pos = np.sign(pred)
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    pnl = pos * act - (cost_bps / 10000.0) * turnover
    mse_model = float(np.mean((pred - act) ** 2))
    mse_zero = float(np.mean(act ** 2))
    d = (pred - act) ** 2 - (0 - act) ** 2
    dm_t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))
    dm_p = float(2 * (1 - stats.norm.cdf(abs(dm_t))))
    sharpe = float(pnl.mean() / (pnl.std(ddof=1) + 1e-12) * np.sqrt(252))
    bh_sharpe = float(act.mean() / (act.std(ddof=1) + 1e-12) * np.sqrt(252))
    hits = int(np.sum(np.sign(pred) == np.sign(act)))
    binom_p = float(stats.binomtest(hits, len(act), 0.5, alternative="greater").pvalue)
    return {
        "n": int(len(pred)),
        "n_folds": int(fold),
        "shuffle_labels": bool(shuffle_labels),
        "directional_accuracy": float(hits / len(act)),
        "binomial_p_accuracy": binom_p,
        "mse_model": mse_model,
        "mse_zero": mse_zero,
        "mse_skill": float(1 - mse_model / mse_zero),
        "dm_stat_vs_zero": dm_t,
        "dm_p_vs_zero": dm_p,
        "sharpe_after_cost": sharpe,
        "mean_daily_pnl": float(pnl.mean()),
        "total_return": float(np.prod(1 + pnl) - 1),
        "buy_hold_total_return": float(np.prod(1 + act) - 1),
        "buy_hold_sharpe": bh_sharpe,
        "n_trades": int(np.sum(turnover > 0)),
        "cost_bps": cost_bps,
        "first_test_date": when[0] if when else None,
        "last_test_date": when[-1] if when else None,
        "equity_model": [float(v) for v in np.cumprod(1 + pnl)],
        "equity_buy_hold": [float(v) for v in np.cumprod(1 + act)],
        "equity_dates": when,
        "preds": [float(v) for v in pred],
        "actual": [float(v) for v in act],
        "pnl": [float(v) for v in pnl],
    }


def volatility_clustering(df: pd.DataFrame, max_lag: int = 10) -> dict:
    r = np.diff(np.log(df["close"].astype(float).values))
    a = np.abs(r)
    acf = {}
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.corrcoef(a[:-lag], a[lag:])[0, 1])
    n = len(a)
    q = n * (n + 2) * sum(acf[k] ** 2 / (n - k) for k in acf)
    p = float(stats.chi2.sf(q, max_lag))
    return {"acf": acf, "ljung_box_q": float(q), "ljung_box_p": p, "n_returns": n,
            "return_vol": float(np.std(r) * np.sqrt(252))}


def _boot_mean(values, n_boot: int = 5000, seed: int = 0):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _boot_sharpe(pnl, n_boot: int = 5000, seed: int = 0):
    a = np.asarray(pnl, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    samples = a[idx]
    sharpes = samples.mean(axis=1) / (samples.std(axis=1, ddof=1) + 1e-12) * np.sqrt(252)
    sharpes_abs = np.abs(sharpes)
    return float(np.abs(a.mean() / (a.std(ddof=1) + 1e-12) * np.sqrt(252))), \
        float(np.quantile(sharpes_abs, 0.025)), float(np.quantile(sharpes_abs, 0.975))


def _boot_acf1(abs_returns, n_boot: int = 5000, seed: int = 0):
    a = np.asarray(abs_returns, dtype=float)
    x, y = a[:-1], a[1:]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if x[idx].std() > 0 and y[idx].std() > 0:
            vals.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    return float(np.corrcoef(x, y)[0, 1]), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run_market_probe(cfg: dict) -> dict:
    df, meta = fetch_prices(
        symbol=cfg.get("symbol", "SPY"), range_=cfg.get("range", "10y"),
        interval=cfg.get("interval", "1d"), cache_dir=cfg.get("cache_dir", "data/cache"),
        max_age_hours=float(cfg.get("max_age_hours", 24)),
    )
    X, y, dates, names = build_features(df, horizon=1)
    kwargs = dict(min_train=int(cfg.get("min_train", 750)),
                  test_size=int(cfg.get("test_size", 63)),
                  embargo=int(cfg.get("embargo", 1)), alpha=10.0,
                  cost_bps=float(cfg.get("cost_bps", 1.0)))
    wf = walk_forward(X, y, dates, **kwargs)
    control = walk_forward(X, y, dates, shuffle_labels=True, **kwargs)
    vol = volatility_clustering(df)
    pred = np.asarray(wf["preds"])
    act = np.asarray(wf["actual"])
    pnl = np.asarray(wf["pnl"])
    hits = (np.sign(pred) == np.sign(act)).astype(float)
    acc, acc_lo, acc_hi = _boot_mean(hits)
    d = (pred - act) ** 2 - act ** 2
    mse_skill = float(1 - np.mean((pred - act) ** 2) / np.mean(act ** 2))
    d_mean, d_lo, d_hi = _boot_mean(d)
    mse_zero = float(np.mean(act ** 2))
    skill_lo, skill_hi = -d_hi / mse_zero, -d_lo / mse_zero
    sharpe_abs, sharpe_lo, sharpe_hi = _boot_sharpe(pnl)
    ret = np.diff(np.log(df["close"].astype(float).values))
    acf1, acf_lo, acf_hi = _boot_acf1(np.abs(ret))
    ctrl_hits = (np.sign(control["preds"]) == np.sign(control["actual"])).astype(float)
    ctrl_acc, ctrl_lo, ctrl_hi = _boot_mean(ctrl_hits)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(ctrl_hits), size=(5000, len(ctrl_hits)))
    ctrl_abs = np.abs(ctrl_hits[idx].mean(axis=1) - 0.5)
    ctrl_dev = abs(ctrl_acc - 0.5)
    ctrl_dev_lo, ctrl_dev_hi = float(np.quantile(ctrl_abs, 0.025)), float(np.quantile(ctrl_abs, 0.975))
    predictions = {
        "M1": {"measured": acc, "ci_low": acc_lo, "ci_high": acc_hi, "n": int(len(hits))},
        "M2": {"measured": mse_skill, "ci_low": skill_lo, "ci_high": skill_hi, "n": int(len(act))},
        "M3": {"measured": sharpe_abs, "ci_low": sharpe_lo, "ci_high": sharpe_hi, "n": int(len(pnl))},
        "M4": {"measured": acf1, "ci_low": acf_lo, "ci_high": acf_hi, "n": int(len(ret) - 1)},
        "M5": {"measured": ctrl_dev, "ci_low": ctrl_dev_lo, "ci_high": ctrl_dev_hi,
               "control_accuracy": ctrl_acc, "n": int(len(control["actual"]))},
    }
    return {
        "meta": meta,
        "n_rows": len(df),
        "features": names,
        "walk_forward": wf,
        "shuffled_control": control,
        "volatility": vol,
        "predictions": predictions,
        "first_date": str(df["date"].iloc[0]),
        "last_date": str(df["date"].iloc[-1]),
    }
