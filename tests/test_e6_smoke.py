from __future__ import annotations

import torch

from ptr.config import ModelConfig
from ptr.experiments.e6_hints import Gate, _eval, _train
from ptr.synthetic import (
    build_hint_table, make_queries, make_update_world, memory_corpus,
    query_prompt, split_corpus,
)
from ptr.tokenizer import WordTokenizer


def test_e6_tiny_end_to_end():
    facts = make_update_world(n_entities=6, n_colors=6, seed=0)
    train_facts, eval_facts, train_lines = split_corpus(
        facts, n_eval=2, templates=(0, 1), repeats=2, seed=0
    )
    memory_lines = memory_corpus(facts, templates=(0, 1, 2))
    text = "the cat sat on the mat. " * 20 + "\n" + "\n".join(train_lines)
    tok = WordTokenizer(
        text + "\n" + "\n".join(memory_lines) + "\n" + "\n".join(
            query_prompt(f, w, 0) for f in facts for w in ("old", "new")
        ),
        min_freq=1, max_vocab=500,
    )
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    table = build_hint_table(memory_lines, tok, order_invariant=False)
    mcfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=2, d_model=16,
                       d_ff=32, ctx=16)
    model = __import__("ptr.model", fromlist=["TinyLM"]).TinyLM(mcfg)
    gate = Gate(16)
    losses = _train(model, gate, ids, table, "always", False, 0, 5, 4, 16, 1e-3)
    assert len(losses) == 5
    out = _eval(model, gate, make_queries(eval_facts, "old", templates=(0,)), tok, table, False, True, 16)
    assert 0.0 <= out["top1"] <= 1.0
    assert out["n"] > 0
