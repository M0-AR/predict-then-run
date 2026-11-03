from __future__ import annotations

import copy

import torch

from ..config import Config
from ..data import sample_batch
from ..model import TinyLM
from ..stats import bootstrap_ci
from ..train import batch_grad_cosine, head_grad_cosine, layer_grad_norms, set_seed, train_model
from .common import char_data, directional_result, save_result, summarise


CONDITIONS = [
    ("baseline", 1, 0.0),
    ("mtp_equal", 2, 1.0),
    ("mtp_downweighted", 2, 0.3),
]


def run(cfg: Config) -> dict:
    ds, manifest = char_data(cfg)
    tok = ds["tokenizer"]
    seeds = list(cfg.seeds)
    runs = []
    for condition, heads, mtp_weight in CONDITIONS:
        mcfg = copy.deepcopy(cfg.model)
        mcfg.vocab_size = tok.vocab_size
        mcfg.mtp_heads = heads
        mcfg.mtp_weight = mtp_weight
        for seed in seeds:
            set_seed(seed)
            model = TinyLM(mcfg)
            res = train_model(
                model, ds["train"], ds["val"], mcfg, cfg.train,
                optimizer_name="adamw", schedule="cosine", seed=seed,
            )
            g = torch.Generator().manual_seed(seed + 4242)
            x, y = sample_batch(ds["val"], cfg.train.batch_size, mcfg.ctx, g)
            bc = batch_grad_cosine(model, ds["train"], cfg.train.batch_size, mcfg.ctx, seed + 11)
            if heads == 1:
                hc = {"head_grad_cosine": float("nan"), "g0_norm": float("nan"), "g1_norm": float("nan")}
                ratio0 = float("nan")
                ratioL = float("nan")
            else:
                hc = head_grad_cosine(model, x, y)
                n0 = layer_grad_norms(model, x, y, head=0)
                n1 = layer_grad_norms(model, x, y, head=1)
                ratio0 = n1[0] / (n0[0] + 1e-12)
                ratioL = n1[-1] / (n0[-1] + 1e-12)
            runs.append({
                "condition": condition, "seed": seed,
                "mtp_heads": heads, "mtp_weight": mtp_weight,
                "val_loss": res.eval["val_loss"],
                "val_loss_h0": res.eval.get("val_loss_h0"),
                "val_loss_h1": res.eval.get("val_loss_h1"),
                "head_grad_cosine": hc["head_grad_cosine"],
                "batch_grad_cosine": bc,
                "ratio_layer0": ratio0,
                "ratio_last": ratioL,
            })
    by_cond = {}
    for condition, _, _ in CONDITIONS:
        sel = [r for r in runs if r["condition"] == condition]
        metrics = ["val_loss_h0", "val_loss_h1", "head_grad_cosine", "batch_grad_cosine",
                   "ratio_layer0", "ratio_last"]
        by_cond[condition] = {
            m: summarise([
                r[m] for r in sel
                if r[m] is not None and not (isinstance(r[m], float) and r[m] != r[m])
            ])
            for m in metrics
        }

    def vals(condition: str, metric: str) -> list[float]:
        return [
            r[metric]
            for r in sorted([r for r in runs if r["condition"] == condition], key=lambda r: r["seed"])
        ]

    predictions = {}
    predictions["E3.P1"] = directional_result(
        vals("baseline", "val_loss_h0"), vals("mtp_equal", "val_loss_h0")
    )
    predictions["E3.P2"] = directional_result(
        vals("mtp_equal", "batch_grad_cosine"), vals("mtp_equal", "head_grad_cosine")
    )
    predictions["E3.P3"] = directional_result(
        vals("mtp_equal", "val_loss_h0"), vals("mtp_downweighted", "val_loss_h0")
    )
    rel = directional_result(
        vals("baseline", "val_loss_h0"), vals("mtp_downweighted", "val_loss_h0")
    )
    predictions["E3.P4"] = {
        "baseline_mean": rel["baseline_mean"],
        "treatment_mean": rel["treatment_mean"],
        "measured": rel["measured_relative"],
        "ci_low": rel["ci_low_relative"],
        "ci_high": rel["ci_high_relative"],
        "measured_relative": rel["measured_relative"],
        "ci_low_relative": rel["ci_low_relative"],
        "ci_high_relative": rel["ci_high_relative"],
        "n": rel["n"],
    }
    cosine_diffs = [
        d - e for e, d in zip(
            vals("mtp_equal", "head_grad_cosine"),
            vals("mtp_downweighted", "head_grad_cosine"),
        )
    ]
    mean, lo, hi = bootstrap_ci(cosine_diffs, seed=0)
    predictions["E3.P5"] = {
        "measured": mean, "ci_low": lo, "ci_high": hi, "n": len(cosine_diffs),
    }
    return {
        "experiment": "e3_mtp",
        "manifest": manifest,
        "config": {
            "seeds": seeds, "steps": cfg.train.steps, "lr": cfg.train.lr,
            "conditions": [{"name": c, "heads": h, "weight": w} for c, h, w in CONDITIONS],
            "model": {k: v for k, v in mcfg.__dict__.items()},
        },
        "runs": runs,
        "by_condition": by_cond,
        "predictions": predictions,
    }


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    payload = run(cfg)
    save_result("e3_mtp", payload)
    return payload
