from __future__ import annotations

import copy

import numpy as np
from scipy import stats

from ..config import Config
from ..model import TinyLM
from ..stats import bootstrap_ci
from ..train import set_seed, train_model
from .common import char_data, directional_result, save_result, summarise


MICRO = [8, 16, 32]
SCHEDULES = ["constant", "cosine_tail"]


def _drop(history: list[dict], decay_start: int) -> tuple[float, float, float]:
    pre = [h["loss"] for h in history if h["step"] <= decay_start][-3:]
    post = [h["loss"] for h in history if h["step"] > decay_start][-3:]
    pre_loss = float(np.mean(pre)) if pre else float("nan")
    post_loss = float(np.mean(post)) if post else float("nan")
    return pre_loss, post_loss, pre_loss - post_loss


def run(cfg: Config) -> dict:
    ds, manifest = char_data(cfg)
    tok = ds["tokenizer"]
    seeds = list(cfg.seeds)
    decay_start = int(0.7 * cfg.train.steps)
    runs = []
    for micro in MICRO:
        for schedule in SCHEDULES:
            for seed in seeds:
                mcfg = copy.deepcopy(cfg.model)
                mcfg.vocab_size = tok.vocab_size
                set_seed(seed)
                model = TinyLM(mcfg)
                res = train_model(
                    model, ds["train"], ds["val"], mcfg, cfg.train,
                    optimizer_name="adamw",
                    schedule="cosine" if schedule == "cosine_tail" else "constant",
                    seed=seed, micro_batch=micro,
                    decay_start=decay_start if schedule == "cosine_tail" else None,
                    noise_probe_step=decay_start, noise_micro_batch=8,
                )
                pre, post, drop = _drop(res.history, decay_start)
                runs.append({
                    "micro_batch": micro, "schedule": schedule, "seed": seed,
                    "pre_loss": pre, "post_loss": post, "drop": drop,
                    "noise_scale": res.noise_scale,
                    "val_loss": res.eval["val_loss"],
                })
    for r in runs:
        constant = next(
            x for x in runs
            if x["micro_batch"] == r["micro_batch"] and x["schedule"] == "constant"
            and x["seed"] == r["seed"]
        )
        r["benefit"] = constant["val_loss"] - r["val_loss"]
    by_cond = {}
    for micro in MICRO:
        for schedule in SCHEDULES:
            sel = [r for r in runs if r["micro_batch"] == micro and r["schedule"] == schedule]
            by_cond[f"{micro}_{schedule}"] = {
                m: summarise([r[m] for r in sel if r[m] is not None])
                for m in ["drop", "benefit", "noise_scale", "val_loss"]
            }

    def get(micro: int, schedule: str) -> list[dict]:
        return sorted(
            [r for r in runs if r["micro_batch"] == micro and r["schedule"] == schedule],
            key=lambda r: r["seed"],
        )

    predictions = {}
    predictions["E5.P1"] = directional_result(
        [r["val_loss"] for r in get(32, "constant")],
        [r["val_loss"] for r in get(32, "cosine_tail")],
    )
    diffs = [
        a["benefit"] - b["benefit"]
        for a, b in zip(get(16, "cosine_tail"), get(32, "cosine_tail"))
    ]
    mean, lo, hi = bootstrap_ci(diffs, seed=0)
    p_diff = float(stats.ttest_1samp(diffs, 0.0).pvalue) if len(diffs) > 1 else float("nan")
    predictions["E5.P2"] = {
        "baseline_mean": float(np.mean([r["benefit"] for r in get(16, "cosine_tail")])),
        "treatment_mean": float(np.mean([r["benefit"] for r in get(32, "cosine_tail")])),
        "measured": mean,
        "ci_low": lo,
        "ci_high": hi,
        "measured_relative": mean,
        "ci_low_relative": lo,
        "ci_high_relative": hi,
        "p_value": p_diff,
        "n": len(diffs),
    }
    cosine = sorted(
        [r for r in runs if r["schedule"] == "cosine_tail" and r["noise_scale"] is not None],
        key=lambda r: (r["micro_batch"], r["seed"]),
    )
    x = np.asarray([r["noise_scale"] for r in cosine], dtype=float)
    y = np.asarray([r["benefit"] for r in cosine], dtype=float)
    if len(cosine) > 3 and x.std() > 0 and y.std() > 0:
        r_corr = float(np.corrcoef(x, y)[0, 1])
        p_corr = float(stats.pearsonr(x, y).pvalue)
    else:
        r_corr, p_corr = 0.0, float("nan")
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(5000):
        idx = rng.integers(0, len(cosine), len(cosine))
        if x[idx].std() > 0 and y[idx].std() > 0:
            rs.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    predictions["E5.P3"] = {
        "measured": r_corr,
        "ci_low": float(np.quantile(rs, 0.025)) if rs else float("nan"),
        "ci_high": float(np.quantile(rs, 0.975)) if rs else float("nan"),
        "p_value": p_corr,
        "n": len(cosine),
    }
    return {
        "experiment": "e5_lr_decay",
        "manifest": manifest,
        "config": {
            "seeds": seeds, "steps": cfg.train.steps, "decay_start": decay_start,
            "micro_batches": MICRO, "model": mcfg.__dict__,
        },
        "runs": runs,
        "by_condition": by_cond,
        "predictions": predictions,
    }


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    payload = run(cfg)
    save_result("e5_lr_decay", payload)
    return payload
