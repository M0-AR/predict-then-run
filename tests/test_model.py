from __future__ import annotations

import torch

from ptr.config import ModelConfig
from ptr.model import TinyLM


def _batch():
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 32, (2, 8), generator=g), torch.randint(0, 32, (2, 8), generator=g)


def test_forward_and_mtp():
    cfg = ModelConfig(vocab_size=32, n_layer=1, n_head=2, d_model=16, d_ff=32, ctx=8, mtp_heads=2)
    model = TinyLM(cfg)
    x, y = _batch()
    hidden, aux = model(x)
    assert hidden.shape == (2, 8, 16)
    loss, heads = model.mtp_loss(hidden, y, aux)
    assert heads.shape[0] == 2
    assert torch.isfinite(loss)
    assert aux == 0.0


def test_moe_forward_and_aux():
    cfg = ModelConfig(vocab_size=32, n_layer=1, n_head=2, d_model=16, d_ff=32, ctx=8,
                      moe_experts=4, moe_top_k=1, aux_loss_weight=0.01)
    model = TinyLM(cfg)
    x, y = _batch()
    hidden, aux = model(x)
    assert hidden.shape == (2, 8, 16)
    assert torch.isfinite(aux)
    assert len(model._router_trace) == cfg.n_layer
