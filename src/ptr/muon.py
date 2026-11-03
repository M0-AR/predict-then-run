from __future__ import annotations

import math

import torch


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    x = x / (x.norm() + eps)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Muon for matrices, AdamW for vectors/scalars (Keller Jordan-style)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if g.ndim == 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(g)
                    update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                    update = update * (0.2 * math.sqrt(max(g.size(0), g.size(1))))
                    if group["weight_decay"]:
                        p.mul_(1 - lr * group["weight_decay"])
                    p.add_(update.to(p.dtype), alpha=-lr)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias1 = 1 - beta1 ** state["step"]
                    bias2 = 1 - beta2 ** state["step"]
                    denom = (v / bias2).sqrt().add_(group["eps"])
                    if group["weight_decay"]:
                        p.mul_(1 - lr * group["weight_decay"])
                    p.add_(m / bias1 / denom, alpha=-lr)
        return loss


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    if name == "muon":
        return Muon(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    raise ValueError(f"unknown optimizer {name}")
