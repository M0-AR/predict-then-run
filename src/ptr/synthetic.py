"""Controlled worlds for the stale-memory experiment.

The experiment separates *external memory* (an n-gram table built from old
facts) from *model knowledge* (facts written into the training corpus).  A
prefix token tells the model whether the memory is supposed to be reliable:

    memory : ent000 color is oldcol03 .
    reason : ent000 color is newcol07 .

The surface prompt after the prefix is identical, so a model can only learn to
gate the memory by using the prefix from its hidden state.  This is the
smallest controlled version of the "learns when to use a hint" claim.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

PREFIX = {"old": "memory", "new": "reason"}


@dataclass(frozen=True)
class UpdateFact:
    entity: str
    old: str
    new: str


def make_update_world(n_entities: int = 80, n_colors: int = 40, seed: int = 0) -> list[UpdateFact]:
    entities = [f"ent{i:03d}" for i in range(n_entities)]
    old_colors = [f"oldcol{i:02d}" for i in range(n_colors)]
    new_colors = [f"newcol{i:02d}" for i in range(n_colors)]
    facts: list[UpdateFact] = []
    for i, entity in enumerate(entities):
        old = old_colors[(i * 7 + 3 + seed) % n_colors]
        new = new_colors[(i * 11 + 5 + seed) % n_colors]
        facts.append(UpdateFact(entity, old, new))
    return facts


def update_sentence(fact: UpdateFact, which: str, template: int) -> str:
    value = fact.old if which == "old" else fact.new
    if template == 0:
        return f"{fact.entity} color is {value} ."
    if template == 1:
        return f"the color of {fact.entity} is {value} ."
    if template == 2:
        return f"{fact.entity} 's color is {value} ."
    raise ValueError(f"unknown template {template}")


def prefixed_sentence(fact: UpdateFact, which: str, template: int, prefix: str | None = None) -> str:
    p = prefix if prefix is not None else PREFIX[which]
    return f"{p} : {update_sentence(fact, which, template)}"


def build_corpus(
    facts: list[UpdateFact],
    templates: tuple[int, ...] = (1, 2),
    repeats: int = 32,
    seed: int = 0,
) -> list[str]:
    """Training corpus: both old and new facts, so the gate sees both regimes."""
    rng = random.Random(seed)
    lines: list[str] = []
    for fact in facts:
        for which in ("old", "new"):
            for template in templates:
                lines.extend([prefixed_sentence(fact, which, template)] * repeats)
    rng.shuffle(lines)
    return lines


def split_corpus(
    facts: list[UpdateFact],
    n_eval: int = 20,
    templates: tuple[int, ...] = (1, 2),
    repeats: int = 32,
    seed: int = 0,
) -> tuple[list[UpdateFact], list[UpdateFact], list[str]]:
    """Train on both facts for most entities, but hold out old facts for eval.

    Held-out entities appear in training only with their *new* facts.  Their
    *old* facts are available only through the external memory table, which is
    what makes the hint useful.  New facts are in the weights, which is what
    makes a stale hint harmful.
    """
    if n_eval >= len(facts):
        raise ValueError("n_eval must leave at least one training entity")
    train_facts = facts[:-n_eval] if n_eval else list(facts)
    eval_facts = facts[-n_eval:] if n_eval else []
    rng = random.Random(seed)
    lines: list[str] = []
    for fact in train_facts:
        for which in ("old", "new"):
            for template in templates:
                lines.extend([prefixed_sentence(fact, which, template)] * repeats)
    for fact in eval_facts:
        for template in templates:
            lines.extend([prefixed_sentence(fact, "new", template)] * repeats)
    rng.shuffle(lines)
    return train_facts, eval_facts, lines


def memory_corpus(facts: list[UpdateFact], templates: tuple[int, ...] = (0, 1, 2)) -> list[str]:
    """External memory: only old facts.  The model never sees these as targets."""
    return [prefixed_sentence(fact, "old", template) for fact in facts for template in templates]


def query_prompt(
    fact: UpdateFact,
    which: str,
    template: int = 0,
    reordered: bool = False,
    prefix: str | None = None,
) -> str:
    p = prefix if prefix is not None else PREFIX[which]
    if reordered:
        if template != 0:
            raise ValueError("reordered prompts are defined for template 0 only")
        return f"{p} : color {fact.entity} is"
    if template == 0:
        return f"{p} : {fact.entity} color is"
    if template == 1:
        return f"{p} : the color of {fact.entity} is"
    if template == 2:
        return f"{p} : {fact.entity} 's color is"
    raise ValueError(f"unknown template {template}")


def make_queries(
    facts: list[UpdateFact],
    which: str,
    templates: tuple[int, ...] = (0,),
    reordered: bool = False,
) -> list[tuple[str, str]]:
    target = "old" if which == "old" else "new"
    return [
        (query_prompt(fact, which, template, reordered), getattr(fact, target))
        for fact in facts
        for template in templates
    ]


def build_hint_table(
    lines: list[str],
    tokenizer,
    order_invariant: bool = False,
    n: int = 3,
) -> dict[tuple[int, ...], tuple[int, float]]:
    """n-gram table: last n tokens -> (most frequent next token, confidence)."""
    counts: dict[tuple[int, ...], dict[int, int]] = {}
    for line in lines:
        ids = tokenizer.encode(line)
        for i in range(n, len(ids)):
            key = tuple(ids[i - n : i])
            if order_invariant:
                key = tuple(sorted(key))
            nxt = ids[i]
            counts.setdefault(key, {})
            counts[key][nxt] = counts[key].get(nxt, 0) + 1
    table: dict[tuple[int, ...], tuple[int, float]] = {}
    for key, nxts in counts.items():
        total = sum(nxts.values())
        nxt, c = max(nxts.items(), key=lambda kv: kv[1])
        table[key] = (nxt, c / total)
    return table


def _lookup_key(ids: list[int], order_invariant: bool, n: int) -> tuple[int, ...]:
    key = tuple(ids[-n:])
    return tuple(sorted(key)) if order_invariant else key


def apply_hint_bias(
    logits,
    input_ids,
    table: dict[tuple[int, ...], tuple[int, float]],
    order_invariant: bool = False,
    gate=None,
    scale: float = 4.0,
    n: int = 3,
):
    """Add the external-memory bias at every matching position.

    ``gate`` is an optional (batch, time, 1) tensor of learned trust values.
    Keeping the gate in the autograd graph is what lets the model learn *when*
    to trust the memory.
    """
    b, t, _ = logits.shape
    bi_l, ti_l, nx_l, vals = [], [], [], []
    for bi in range(b):
        row = input_ids[bi].tolist()
        for ti in range(n, t):
            key = tuple(row[ti - n : ti])
            if order_invariant:
                key = tuple(sorted(key))
            item = table.get(key)
            if item is None:
                continue
            nxt, weight = item
            bi_l.append(bi)
            ti_l.append(ti)
            nx_l.append(nxt)
            if gate is not None:
                vals.append(gate[bi, ti, 0] * (weight * scale))
            else:
                vals.append(weight * scale)
    if not vals:
        return logits
    idx = (torch.tensor(bi_l), torch.tensor(ti_l), torch.tensor(nx_l))
    if gate is not None:
        vals_t = torch.stack(vals)
    else:
        vals_t = torch.tensor(vals, dtype=logits.dtype, device=logits.device)
    return logits.index_put(idx, vals_t, accumulate=True)


def apply_hint_bias_last(
    logits,
    input_ids,
    table: dict[tuple[int, ...], tuple[int, float]],
    order_invariant: bool = False,
    gate=None,
    scale: float = 4.0,
    n: int = 3,
):
    """Add the external-memory bias at the final position only."""
    out = logits.clone()
    for bi in range(out.size(0)):
        key = _lookup_key(input_ids[bi].tolist(), order_invariant, n)
        item = table.get(key)
        if item is None:
            continue
        nxt, weight = item
        if gate is not None:
            out[bi, nxt] = out[bi, nxt] + gate[bi, 0] * weight * scale
        else:
            out[bi, nxt] = out[bi, nxt] + weight * scale
    return out
