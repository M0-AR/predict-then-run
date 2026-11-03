from __future__ import annotations

import copy

from ..config import Config
from ..model import TinyLM
from ..train import set_seed, train_model
from .common import char_data, directional_result, save_result, summarise


def run(cfg: Config) -> dict:
    ds, manifest = char_data(cfg)
    tok = ds["tokenizer"]
    mcfg = copy.deepcopy(cfg.model)
    mcfg.vocab_size = tok.vocab_size
    conditions = [0.0, 0.1, 0.5]
    runs = []
    for wd in conditions:
        for seed in cfg.seeds:
            set_seed(seed)
            model = TinyLM(mcfg)
            tcfg = copy.deepcopy(cfg.train)
            tcfg.weight_decay = wd
            res = train_model(
                model, ds["train"], ds["val"], mcfg, tcfg,
                optimizer_name="adamw", schedule="cosine", seed=seed,
            )
            runs.append({
                "wd": wd,
                "seed": seed,
                "val_loss": res.eval["val_loss"],
                "weight_norm": res.weight_norm,
                "rel_weight_change": res.rel_weight_change,
                "grad_norm": res.final_grad_norm,
                "elr": res.elr,
                "lr": res.final_lr,
            })
    by_cond = {}
    for wd in conditions:
        sel = [r for r in runs if r["wd"] == wd]
        by_cond[str(wd)] = {
            m: summarise([r[m] for r in sel]) for m in
            ["val_loss", "weight_norm", "rel_weight_change", "grad_norm", "elr"]
        }
    base = [r for r in runs if r["wd"] == 0.0]
    treat = [r for r in runs if r["wd"] == 0.1]
    base.sort(key=lambda r: r["seed"])
    treat.sort(key=lambda r: r["seed"])
    predictions = {}
    for metric, pid in [
        ("weight_norm", "E1.P1"),
        ("rel_weight_change", "E1.P2"),
        ("val_loss", "E1.P3"),
        ("elr", "E1.P4"),
    ]:
        predictions[pid] = directional_result(
            [r[metric] for r in base], [r[metric] for r in treat]
        )
    return {
        "experiment": "e1_weight_decay",
        "manifest": manifest,
        "config": {"seeds": cfg.seeds, "steps": cfg.train.steps, "lr": cfg.train.lr,
                   "model": mcfg.__dict__, "conditions": conditions},
        "runs": runs,
        "by_condition": by_cond,
        "predictions": predictions,
    }


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    payload = run(cfg)
    save_result("e1_weight_decay", payload)
    return payload
