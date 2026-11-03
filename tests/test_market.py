from __future__ import annotations

import numpy as np
import pandas as pd

from ptr.market import build_features, volatility_clustering, walk_forward


def _prices(n=400):
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "close": close})


def test_features_are_causal_shapes():
    df = _prices()
    X, y, dates, names = build_features(df)
    assert X.shape[1] == len(names) == 5
    assert len(X) == len(y) == len(dates)
    assert np.isfinite(X).all() and np.isfinite(y).all()


def test_walk_forward_outputs_and_control():
    df = _prices(500)
    X, y, dates, _ = build_features(df)
    out = walk_forward(X, y, dates, min_train=200, test_size=50)
    assert out["n"] > 0
    assert 0 <= out["directional_accuracy"] <= 1
    assert "sharpe_after_cost" in out
    ctrl = walk_forward(X, y, dates, min_train=200, test_size=50, shuffle_labels=True)
    assert ctrl["n"] == out["n"]


def test_volatility_clustering():
    out = volatility_clustering(_prices())
    assert 1 in out["acf"]
    assert np.isfinite(out["ljung_box_q"])
