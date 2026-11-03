from __future__ import annotations

import copy

from ..config import Config
from ..model import TinyLM
from ..stats import bootstrap_ci
from ..train import set_seed, train_model
from .common import char_data, directional_result, save_result, summarise


def _one(cfg: Config, ds: dict, tok, width: int, opt: str, lr: float, seed: int, steps: int) -> dict:
    mcfg = copy.deepcopy(cfg.model)
    mcfg.vocab_size = tok.vocab_size
    mcfg.d_model = width
    mcfg.d_ff = 4 * width
    tcfg = copy.deepcopy(cfg.train)
    tcfg.lr = lr
    tcfg.steps = steps
    set_seed(seed)
    model = TinyLM(mcfg)
    res = train_model(
        model, ds["train"], ds["val"], mcfg, tcfg,
        optimizer_name=opt, schedule="cosine", seed=seed,
    )
    return {
        "width": width,
        "optimizer": opt,
        "lr": lr,
        "seed": seed,
        "steps": steps,
        "val_loss": res.eval["val_loss"],
        "final_train_loss": res.history[-1]["loss"] if res.history else None,
    }


def run(cfg: Config) -> dict:
    ds, manifest = char_data(cfg)
    tok = ds["tokenizer"]
    widths = [64, 128]
    tune_steps = min(100, cfg.train.steps)

    tuning = []
    for width in widths:
        for opt in ["adamw", "muon"]:
            lrs = [0.001, 0.003, 0.01, 0.02] if opt == "adamw" else [0.003, 0.01, 0.02, 0.04, 0.06]
            for lr in lrs:
                tuning.append(_one(cfg, ds, tok, width, opt, lr, 0, tune_steps))

    best = {}
    for width in widths:
        for opt in ["adamw", "muon"]:
            rows = [r for r in tuning if r["width"] == width and r["optimizer"] == opt]
            best[(width, opt)] = min(rows, key=lambda r: r["val_loss"])

    runs = []
    for width in widths:
        for opt in ["adamw", "muon"]:
            lr = best[(width, opt)]["lr"]
            for seed in cfg.seeds:
                runs.append(_one(cfg, ds, tok, width, opt, lr, seed, cfg.train.steps))

    by_cond = {}
    for width in widths:
        for opt in ["adamw", "muon"]:
            sel = sorted(
                [r for r in runs if r["width"] == width and r["optimizer"] == opt],
                key=lambda r: r["seed"],
            )
            by_cond[f"{width}_{opt}"] = {
                "lr": sel[0]["lr"],
                **{
                    m: summarise([r[m] for r in sel if r[m] is not None])
                    for m in ["val_loss", "final_train_loss"]
                },
            }

    def losses(width: int, opt: str) -> list[float]:
        return [
            r["val_loss"]
            for r in sorted(
                [r for r in runs if r["width"] == width and r["optimizer"] == opt],
                key=lambda r: r["seed"],
            )
        ]

    predictions = {}
    predictions["E2.P1"] = directional_result(losses(64, "adamw"), losses(64, "muon"))
    adv64 = [a - m for a, m in zip(losses(64, "adamw"), losses(64, "muon"))]
    adv128 = [a - m for a, m in zip(losses(128, "adamw"), losses(128, "muon"))]
    diffs = [b - a for a, b in zip(adv64, adv128)]
    mean, lo, hi = bootstrap_ci(diffs, seed=0)
    predictions["E2.P2"] = {
        "baseline_mean": float(sum(adv64) / len(adv64)),
        "treatment_mean": float(sum(adv128) / len(adv128)),
        "measured_relative": mean,
        "ci_low_relative": lo,
        "ci_high_relative": hi,
        "absolute_diff": mean,
        "absolute_ci_low": lo,
        "absolute_ci_high": hi,
        "n": len(diffs),
        "unit": "loss_difference_of_advantage",
    }
    rel = [
        (m - a) / a
        for a, m in zip(losses(128, "adamw"), losses(128, "muon"))
    ]
    mean, lo, hi = bootstrap_ci(rel, seed=0)
    predictions["E2.P3"] = {
        "baseline_mean": 0.0,
        "treatment_mean": float(sum(rel) / len(rel)),
        "measured": float(sum(rel) / len(rel)),
        "ci_low": lo,
        "ci_high": hi,
        "measured_relative": float(sum(rel) / len(rel)),
        "ci_low_relative": lo,
        "ci_high_relative": hi,
        "n": len(rel),
    }
    return {
        "experiment": "e2_muon",
        "manifest": manifest,
        "config": {
            "seeds": list(cfg.seeds), "steps": cfg.train.steps, "tune_steps": tune_steps,
            "widths": widths, "model": cfg.model.__dict__,
        },
        "tuning": tuning,
        "best_lr": {f"{w}_{o}": best[(w, o)]["lr"] for w in widths for o in ["adamw", "muon"]},
        "runs": runs,
        "by_condition": by_cond,
        "predictions": predictions,
    }


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    payload = run(cfg)
    save_result("e2_muon", payload)
    return payload
