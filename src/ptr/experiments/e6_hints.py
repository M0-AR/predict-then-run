"""E6: does a model learn *when* to trust an external memory?

The memory table contains old facts.  The training corpus contains both old
facts (prefix ``memory``) and updated facts (prefix ``reason``).  The same
surface prompt follows the prefix, so the only way to use the memory correctly
is to condition a learned gate on the prefix.  This is a controlled version of
the claim "the model will learn to turn hints off when it needs to reason".
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data import load_public_text, sample_batch
from ..model import TinyLM
from ..stats import bootstrap_ci
from ..synthetic import (
    apply_hint_bias,
    apply_hint_bias_last,
    build_hint_table,
    make_queries,
    make_update_world,
    memory_corpus,
    query_prompt,
    split_corpus,
)
from ..tokenizer import WordTokenizer
from ..train import set_seed
from .common import rate_result, save_result, summarise

HINT_SCALE = 4.0


class Gate(nn.Module):
    """Per-token trust gate over the external memory."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, 2.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(hidden))


def _train(
    model: TinyLM,
    gate: Gate | None,
    ids: torch.Tensor,
    table,
    hints_mode: str,
    order_invariant: bool,
    seed: int,
    steps: int,
    batch: int,
    ctx: int,
    lr: float,
) -> list[float]:
    set_seed(seed)
    model.train()
    if gate is not None:
        gate.train()
    params = list(model.parameters()) + (list(gate.parameters()) if gate is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(seed + 123)
    rng = random.Random(seed)
    losses: list[float] = []
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = sample_batch(ids, batch, ctx, g)
        hidden, _ = model(x)
        logits = model.heads[0](hidden)
        use_hints = hints_mode in ("always", "bow") or (
            hints_mode == "mixed" and rng.random() < 0.5
        )
        if use_hints and table is not None:
            gate_values = gate(hidden) if gate is not None else None
            logits = apply_hint_bias(
                logits, x, table, order_invariant, gate_values, HINT_SCALE
            )
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def _eval(
    model: TinyLM,
    gate: Gate | None,
    queries: list[tuple[str, str]],
    tok: WordTokenizer,
    table,
    order_invariant: bool,
    hints_on: bool,
    ctx: int,
) -> dict:
    model.eval()
    if gate is not None:
        gate.eval()
    top1 = 0
    mrr = 0.0
    loss_sum = 0.0
    gates: list[float] = []
    n = 0
    for prompt, target in queries:
        ids = tok.encode(prompt)[-ctx:]
        if not ids:
            continue
        tid = tok.stoi.get(target, tok.unk_id)
        x = torch.tensor([ids], dtype=torch.long)
        hidden, _ = model(x)
        logits = model.heads[0](hidden[:, -1])
        if hints_on and table is not None:
            gate_values = gate(hidden[:, -1]) if gate is not None else None
            if gate_values is not None:
                gates.append(float(gate_values.mean()))
            logits = apply_hint_bias_last(
                logits, x, table, order_invariant, gate_values, HINT_SCALE
            )
        loss = F.cross_entropy(logits, torch.tensor([tid]))
        rank = int((logits[0] > logits[0, tid]).sum().item()) + 1
        top1 += int(rank == 1)
        mrr += 1.0 / rank
        loss_sum += float(loss)
        n += 1
    model.train()
    if gate is not None:
        gate.train()
    return {
        "top1": top1 / max(1, n),
        "mrr": mrr / max(1, n),
        "loss": loss_sum / max(1, n),
        "n": n,
        "gate_mean": float(np.mean(gates)) if gates else None,
    }


def run(cfg: Config) -> dict:
    e6 = cfg.data.get("e6", {})
    natural_chars = int(e6.get("natural_chars", 50_000))
    natural, manifest = load_public_text(
        "tiny_stories",
        cache_dir=cfg.data.get("cache_dir", "data/cache"),
        max_chars=natural_chars,
        url=cfg.data.get("tiny_stories_url"),
    )
    facts = make_update_world(
        n_entities=int(e6.get("n_entities", 80)),
        n_colors=int(e6.get("n_colors", 40)),
        seed=0,
    )
    n_eval = int(e6.get("n_eval_entities", 20))
    train_facts, eval_facts, train_lines = split_corpus(
        facts,
        n_eval=n_eval,
        templates=tuple(e6.get("train_templates", (1, 2))),
        repeats=int(e6.get("repeats", 32)),
        seed=0,
    )
    memory_lines = memory_corpus(facts, templates=(0, 1, 2))
    text = natural + "\n" + "\n".join(train_lines)
    tokenizer_text = (
        text
        + "\n"
        + "\n".join(memory_lines)
        + "\n"
        + "\n".join(query_prompt(f, w, 0) for f in facts for w in ("old", "new"))
    )
    tok = WordTokenizer(tokenizer_text, min_freq=1, max_vocab=8000)
    train_ids = torch.tensor(tok.encode(text), dtype=torch.long)
    table_tri = build_hint_table(memory_lines, tok, order_invariant=False)
    table_bow = build_hint_table(memory_lines, tok, order_invariant=True)

    old_canon = make_queries(eval_facts, "old", templates=(0,))
    new_canon = make_queries(eval_facts, "new", templates=(0,))
    old_reorder = make_queries(eval_facts, "old", templates=(0,), reordered=True)
    new_reorder = make_queries(eval_facts, "new", templates=(0,), reordered=True)
    old_indist = make_queries(eval_facts, "old", templates=(1, 2))
    new_indist = make_queries(eval_facts, "new", templates=(1, 2))

    conditions = {
        "no_hints": ("none", None, False, False),
        "always_hints": ("always", table_tri, False, False),
        "mixed_hints": ("mixed", table_tri, False, False),
        "gated_hints": ("always", table_tri, False, True),
        "bow_gated": ("always", table_bow, True, True),
    }
    seeds = [int(s) for s in e6.get("seeds", cfg.seeds[:3])]
    steps = int(e6.get("steps", cfg.train.steps))
    batch = int(e6.get("batch_size", cfg.train.batch_size))
    runs: list[dict] = []
    for condition, (hints_mode, table, order_invariant, gated) in conditions.items():
        for seed in seeds:
            mcfg = copy.deepcopy(cfg.model)
            mcfg.vocab_size = tok.vocab_size
            mcfg.d_model = int(e6.get("d_model", 64))
            mcfg.d_ff = int(e6.get("d_ff", 256))
            mcfg.ctx = int(e6.get("ctx", 48))
            set_seed(seed)
            model = TinyLM(mcfg)
            gate = Gate(mcfg.d_model) if gated else None
            losses = _train(
                model,
                gate,
                train_ids,
                table,
                hints_mode,
                order_invariant,
                seed,
                steps,
                batch,
                mcfg.ctx,
                cfg.train.lr,
            )
            row = {
                "condition": condition,
                "seed": seed,
                "train_loss": float(np.mean(losses[-20:])),
                "old_canon_on": _eval(model, gate, old_canon, tok, table, order_invariant, True, mcfg.ctx),
                "old_canon_off": _eval(model, gate, old_canon, tok, table, order_invariant, False, mcfg.ctx),
                "new_canon_on": _eval(model, gate, new_canon, tok, table, order_invariant, True, mcfg.ctx),
                "new_canon_off": _eval(model, gate, new_canon, tok, table, order_invariant, False, mcfg.ctx),
                "old_reorder_on": _eval(model, gate, old_reorder, tok, table, order_invariant, True, mcfg.ctx),
                "old_reorder_off": _eval(model, gate, old_reorder, tok, table, order_invariant, False, mcfg.ctx),
                "new_reorder_on": _eval(model, gate, new_reorder, tok, table, order_invariant, True, mcfg.ctx),
                "new_reorder_off": _eval(model, gate, new_reorder, tok, table, order_invariant, False, mcfg.ctx),
                "old_indist_on": _eval(model, gate, old_indist, tok, table, order_invariant, True, mcfg.ctx),
                "old_indist_off": _eval(model, gate, old_indist, tok, table, order_invariant, False, mcfg.ctx),
                "new_indist_on": _eval(model, gate, new_indist, tok, table, order_invariant, True, mcfg.ctx),
                "new_indist_off": _eval(model, gate, new_indist, tok, table, order_invariant, False, mcfg.ctx),
                "n_train_tokens": int(train_ids.numel()),
                "vocab_size": tok.vocab_size,
                "table_size": len(table) if table else 0,
            }
            runs.append(row)

    by_cond = {}
    metric_paths = [
        "old_canon_on", "old_canon_off", "new_canon_on", "new_canon_off",
        "old_reorder_on", "old_reorder_off", "new_reorder_on", "new_reorder_off",
        "old_indist_on", "old_indist_off", "new_indist_on", "new_indist_off",
    ]
    for condition in conditions:
        selected = [r for r in runs if r["condition"] == condition]
        summary = {"train_loss": summarise([r["train_loss"] for r in selected])}
        for path in metric_paths:
            summary[f"{path}_top1"] = summarise([r[path]["top1"] for r in selected])
            summary[f"{path}_loss"] = summarise([r[path]["loss"] for r in selected])
        gates_old = [r["old_canon_on"]["gate_mean"] for r in selected if r["old_canon_on"]["gate_mean"] is not None]
        gates_new = [r["new_canon_on"]["gate_mean"] for r in selected if r["new_canon_on"]["gate_mean"] is not None]
        summary["gate_old"] = summarise(gates_old) if gates_old else None
        summary["gate_new"] = summarise(gates_new) if gates_new else None
        by_cond[condition] = summary

    def ordered(condition: str) -> list[dict]:
        return sorted([r for r in runs if r["condition"] == condition], key=lambda r: r["seed"])

    gated = ordered("gated_hints")

    predictions = {}

    def rate(condition: str, family: str, hints: str) -> list[float]:
        return [r[f"{family}_{hints}"]["top1"] for r in ordered(condition)]

    # P1: external memory recovers held-out OLD facts that are absent from weights.
    predictions["E6.P1"] = rate_result(
        rate("no_hints", "old_canon", "off"),
        rate("always_hints", "old_canon", "on"),
    )
    # P2: the same memory is stale for NEW facts and interferes with them.
    predictions["E6.P2"] = rate_result(
        rate("no_hints", "new_canon", "off"),
        rate("always_hints", "new_canon", "on"),
    )
    # P3: a learned gate recovers NEW-fact accuracy under the stale hint.
    predictions["E6.P3"] = rate_result(
        rate("always_hints", "new_canon", "on"),
        rate("gated_hints", "new_canon", "on"),
    )
    # P4: the gate opens more for OLD (memory) prompts than for NEW (reason) prompts.
    gate_diffs = [
        r["old_canon_on"]["gate_mean"] - r["new_canon_on"]["gate_mean"]
        for r in gated
        if r["old_canon_on"]["gate_mean"] is not None and r["new_canon_on"]["gate_mean"] is not None
    ]
    gate_mean, gate_lo, gate_hi = bootstrap_ci(gate_diffs, seed=0)
    predictions["E6.P4"] = {
        "baseline_mean": 0.0,
        "treatment_mean": gate_mean,
        "measured_relative": gate_mean,
        "ci_low_relative": gate_lo,
        "ci_high_relative": gate_hi,
        "absolute_diff": gate_mean,
        "absolute_ci_low": gate_lo,
        "absolute_ci_high": gate_hi,
        "n": len(gate_diffs),
        "unit": "gate_probability_difference",
    }
    # P5: order-invariant keys recover reordered OLD queries.
    predictions["E6.P5"] = rate_result(
        rate("gated_hints", "old_reorder", "on"),
        rate("bow_gated", "old_reorder", "on"),
    )
    # P6: even with gating, memory still helps OLD facts vs no memory at all.
    predictions["E6.P6"] = rate_result(
        rate("no_hints", "old_canon", "off"),
        rate("gated_hints", "old_canon", "on"),
    )
    # P7: naive hint dropout (the transcript's proposed fix) recovers NEW facts?
    predictions["E6.P7"] = rate_result(
        rate("always_hints", "new_canon", "on"),
        rate("mixed_hints", "new_canon", "on"),
    )
    # P8: even the learned gate leaves a residual stale-memory penalty on NEW facts.
    predictions["E6.P8"] = rate_result(
        rate("gated_hints", "new_canon", "off"),
        rate("gated_hints", "new_canon", "on"),
    )
    # P9: the learned gate beats naive hint dropout on NEW facts when hints are on.
    predictions["E6.P9"] = rate_result(
        rate("mixed_hints", "new_canon", "on"),
        rate("gated_hints", "new_canon", "on"),
    )
    return {
        "experiment": "e6_hints",
        "manifest": manifest,
        "config": {
            "seeds": seeds,
            "steps": steps,
            "batch_size": batch,
            "ctx": int(e6.get("ctx", 48)),
            "d_model": int(e6.get("d_model", 64)),
            "n_entities": len(facts),
            "n_train_entities": len(train_facts),
            "n_eval_entities": len(eval_facts),
            "train_templates": list(e6.get("train_templates", (1, 2))),
            "eval_templates": [0],
            "repeats": int(e6.get("repeats", 32)),
            "hint_scale": HINT_SCALE,
            "conditions": list(conditions.keys()),
        },
        "runs": runs,
        "by_condition": by_cond,
        "predictions": predictions,
    }


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    payload = run(cfg)
    save_result("e6_hints", payload)
    return payload
