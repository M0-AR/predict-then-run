from __future__ import annotations

import copy
import math

import numpy as np

from ..config import Config
from ..data import sample_batch
from ..model import TinyLM
from ..train import collect_expert_load, set_seed, train_model
from .common import char_data, directional_result, save_result, summarise


def _early_late_corr(early: list[list[float]], late: list[list[float]]) -> float:
    rs = []
    for a, b in zip(early, late):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.std() == 0 or b.std() == 0:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")


def run(cfg: Config) -> dict:
    ds, manifest = char_data(cfg)
    tok = ds["tokenizer"]
    seeds = cfg.seeds[:3]
    probe_batches = []
    for i in range(4):
        probe_batches.append(sample_batch(ds["train"], cfg.train.batch_size, cfg.model.ctx,
                                           __import__("torch").Generator().manual_seed(100 + i)))
    final_batches = probe_batches
    runs = []
    for aux in [0.0, 0.01]:
        for seed in seeds:
            mcfg = copy.deepcopy(cfg.model)
            mcfg.vocab_size = tok.vocab_size
            mcfg.moe_experts = 4
            mcfg.moe_top_k = 1
            mcfg.aux_loss_weight = aux
            set_seed(seed)
            model = TinyLM(mcfg)
            res = train_model(
                model, ds["train"], ds["val"], mcfg, cfg.train,
                optimizer_name="adamw", schedule="cosine", seed=seed,
                probe_steps=[50], probe_batches=probe_batches,
            )
            final_load = collect_expert_load(model, final_batches)
            early = res.router_probe or {"per_layer": []}
            corr = _early_late_corr(
                [p["probs"] for p in early.get("per_layer", [])],
                [p["probs"] for p in final_load.get("per_layer", [])],
            )
            runs.append({
                "aux": aux, "seed": seed,
                "val_loss": res.eval["val_loss"],
                "entropy": final_load["entropy"],
                "entropy_norm": final_load["entropy"] / math.log(4),
                "max_share": final_load["max_share"],
                "early_late_corr": corr,
                "early_probs": [p["probs"] for p in early.get("per_layer", [])],
                "final_probs": [p["probs"] for p in final_load.get("per_layer", [])],
            })
    by_cond = {}
    for aux in [0.0, 0.01]:
        sel = [r for r in runs if r["aux"] == aux]
        by_cond[str(aux)] = {
            m: summarise([r[m] for r in sel]) for m in
            ["val_loss", "entropy", "entropy_norm", "max_share", "early_late_corr"]
        }
    none = sorted([r for r in runs if r["aux"] == 0.0], key=lambda r: r["seed"])
    with_aux = sorted([r for r in runs if r["aux"] == 0.01], key=lambda r: r["seed"])
    ent_none = summarise([r["entropy_norm"] for r in none])
    corr_none = summarise([r["early_late_corr"] for r in none])
    predictions = {
        "E4.P1": {
            "measured": ent_none["mean"], "ci_low": ent_none["ci_low"],
            "ci_high": ent_none["ci_high"], "n": ent_none["n"],
        },
        "E4.P2": directional_result([r["entropy_norm"] for r in none], [r["entropy_norm"] for r in with_aux]),
        "E4.P3": {
            "measured": corr_none["mean"], "ci_low": corr_none["ci_low"],
            "ci_high": corr_none["ci_high"], "n": corr_none["n"],
        },
        "E4.P4": directional_result([r["val_loss"] for r in none], [r["val_loss"] for r in with_aux]),
    }
    return {
        "experiment": "e4_moe",
        "manifest": manifest,
        "config": {"seeds": seeds, "steps": cfg.train.steps, "experts": 4, "top_k": 1,
                   "aux_weights": [0.0, 0.01], "model": mcfg.__dict__},
        "runs": runs,
        "by_condition": by_cond,
        "predictions": predictions,
    }


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    payload = run(cfg)
    save_result("e4_moe", payload)
    return payload
