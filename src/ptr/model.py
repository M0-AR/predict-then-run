from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.moe_experts
        self.top_k = min(cfg.moe_top_k, cfg.moe_experts)
        self.router = nn.Linear(cfg.d_model, cfg.moe_experts, bias=False)
        self.experts = nn.ModuleList([MLP(cfg) for _ in range(cfg.moe_experts)])
        self.last_probs: torch.Tensor | None = None
        self.last_top1: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = topk_idx == e
            token_mask = mask.any(dim=-1)
            if token_mask.any():
                weight = (topk_probs * mask).sum(dim=-1, keepdim=True)
                out[token_mask] += weight[token_mask] * self.experts[e](flat[token_mask])
        frac = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            frac[e] = (topk_idx == e).float().mean()
        mean_prob = probs.mean(dim=0)
        aux = self.n_experts * (frac * mean_prob).sum()
        self.last_probs = probs.detach()
        self.last_top1 = topk_idx[:, 0].detach()
        return out.view(b, t, d), aux


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.use_moe = cfg.moe_experts > 0
        self.ff = MoE(cfg) if self.use_moe else MLP(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        if self.use_moe:
            h, aux = self.ff(self.ln2(x))
            return x + h, aux
        return x + self.ff(self.ln2(x)), x.new_zeros(())


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) for _ in range(cfg.mtp_heads)]
        )
        self._router_trace: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.apply(self._init)
        if cfg.moe_experts > 0:
            for block in self.blocks:
                if block.use_moe:
                    nn.init.normal_(block.ff.router.weight, mean=0.0, std=cfg.router_init_std)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = idx.shape
        if t > self.cfg.ctx:
            raise ValueError(f"sequence length {t} exceeds context {self.cfg.ctx}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        aux_total = x.new_zeros(())
        self._router_trace = []
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux
            if block.use_moe:
                assert block.ff.last_probs is not None and block.ff.last_top1 is not None
                self._router_trace.append((block.ff.last_probs, block.ff.last_top1))
        return self.ln_f(x), aux_total

    def mtp_loss(self, hidden: torch.Tensor, targets: torch.Tensor, aux_loss: torch.Tensor):
        losses = []
        for h, head in enumerate(self.heads):
            if hidden.size(1) - h <= 0:
                continue
            logits = head(hidden[:, : hidden.size(1) - h])
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, h:].reshape(-1)))
        total = losses[0]
        for h in range(1, len(losses)):
            total = total + self.cfg.mtp_weight * losses[h]
        total = total + self.cfg.aux_loss_weight * aux_loss
        return total, torch.stack(losses)
