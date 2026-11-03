from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .data import sample_batch
from .model import TinyLM
from .muon import make_optimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, cfg: TrainConfig, schedule: str = "cosine") -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if schedule == "constant":
        return cfg.lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


def lr_at_tail(step: int, cfg: TrainConfig, decay_start: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step < decay_start:
        return cfg.lr
    progress = (step - decay_start) / max(1, cfg.steps - decay_start)
    progress = min(1.0, max(0.0, progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def matrix_params(model: TinyLM, exclude_embedding: bool = True):
    out = []
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        if exclude_embedding and (name.startswith("tok") or name.startswith("pos")):
            continue
        out.append((name, p))
    return out


@torch.no_grad()
def param_norm(model: TinyLM) -> float:
    ps = [p.norm().item() for _, p in matrix_params(model)]
    return float(np.mean(ps)) if ps else 0.0


def grad_norm(model: TinyLM) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def relative_weight_change(model: TinyLM, initial: dict[str, torch.Tensor]) -> float:
    num = 0.0
    den = 0.0
    for name, p in matrix_params(model):
        w0 = initial[name]
        num += (p.detach() - w0).pow(2).sum().item()
        den += w0.pow(2).sum().item()
    return math.sqrt(num) / (math.sqrt(den) + 1e-12)


@torch.no_grad()
def evaluate(model: TinyLM, val_ids: torch.Tensor, batch_size: int, ctx: int,
             n_batches: int = 16, seed: int = 1234) -> dict[str, float]:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses = [[] for _ in range(len(model.heads))]
    auxes = []
    for _ in range(n_batches):
        x, y = sample_batch(val_ids, batch_size, ctx, g)
        hidden, aux = model(x)
        for h, head in enumerate(model.heads):
            if hidden.size(1) - h <= 0:
                continue
            logits = head(hidden[:, : hidden.size(1) - h])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:, h:].reshape(-1))
            losses[h].append(loss.item())
        auxes.append(float(aux))
    model.train()
    out = {f"val_loss_h{h}": float(np.mean(v)) for h, v in enumerate(losses) if v}
    out["val_aux"] = float(np.mean(auxes))
    out["val_loss"] = out.get("val_loss_h0", float("nan"))
    return out


@torch.no_grad()
def collect_expert_load(model: TinyLM, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    model.eval()
    counts = None
    for x, _ in batches:
        model(x)
        trace = model._router_trace
        if counts is None:
            counts = [torch.zeros(p.shape[-1]) for p, _ in trace]
        for layer, (_, top1) in enumerate(trace):
            counts[layer] += torch.bincount(top1.cpu(), minlength=counts[layer].numel()).float()
    per_layer = []
    for layer, c in enumerate(counts or []):
        probs = c / c.sum()
        e = probs.numel()
        entropy = float(-(probs * (probs + 1e-12).log()).sum() / math.log(e)) if e > 1 else 1.0
        per_layer.append({
            "probs": probs.tolist(),
            "max_share": float(probs.max()),
            "entropy": entropy,
        })
    model.train()
    if not per_layer:
        return {"max_share": float("nan"), "entropy": float("nan"), "probs": []}
    return {
        "max_share": float(np.mean([p["max_share"] for p in per_layer])),
        "entropy": float(np.mean([p["entropy"] for p in per_layer])),
        "probs": [p["probs"] for p in per_layer],
        "per_layer": per_layer,
    }


def grad_vector(model: TinyLM, loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def head_grad_cosine(model: TinyLM, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    trunk = [p for n, p in model.named_parameters() if not n.startswith("heads.")]
    model.zero_grad(set_to_none=True)
    hidden, aux = model(x)
    loss0 = F.cross_entropy(model.heads[0](hidden).reshape(-1, model.heads[0].out_features), y.reshape(-1))
    g0 = grad_vector(model, loss0, trunk)
    model.zero_grad(set_to_none=True)
    if len(model.heads) > 1:
        hidden, aux = model(x)
        h = 1
        logits = model.heads[h](hidden[:, : hidden.size(1) - h])
        loss1 = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:, h:].reshape(-1))
        g1 = grad_vector(model, loss1, trunk)
        cos = torch.nn.functional.cosine_similarity(g0, g1, dim=0).item()
    else:
        cos = float("nan")
    model.zero_grad(set_to_none=True)
    return {"head_grad_cosine": cos, "g0_norm": g0.norm().item(),
            "g1_norm": float(g1.norm().item()) if len(model.heads) > 1 else float("nan")}


def batch_grad_cosine(model: TinyLM, ids: torch.Tensor, batch_size: int, ctx: int, seed: int) -> float:
    g = torch.Generator().manual_seed(seed)
    trunk = [p for n, p in model.named_parameters() if not n.startswith("heads.")]
    cosines = []
    for _ in range(4):
        x1, y1 = sample_batch(ids, batch_size, ctx, g)
        x2, y2 = sample_batch(ids, batch_size, ctx, g)
        model.zero_grad(set_to_none=True)
        hidden, _ = model(x1)
        l1 = F.cross_entropy(model.heads[0](hidden).reshape(-1, model.heads[0].out_features), y1.reshape(-1))
        g1 = grad_vector(model, l1, trunk)
        model.zero_grad(set_to_none=True)
        hidden, _ = model(x2)
        l2 = F.cross_entropy(model.heads[0](hidden).reshape(-1, model.heads[0].out_features), y2.reshape(-1))
        g2 = grad_vector(model, l2, trunk)
        cosines.append(torch.nn.functional.cosine_similarity(g1, g2, dim=0).item())
    model.zero_grad(set_to_none=True)
    return float(np.mean(cosines))


def layer_grad_norms(model: TinyLM, x: torch.Tensor, y: torch.Tensor, head: int) -> list[float]:
    model.zero_grad(set_to_none=True)
    hidden, _ = model(x)
    logits = model.heads[head](hidden[:, : hidden.size(1) - head])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:, head:].reshape(-1))
    loss.backward()
    out = []
    for block in model.blocks:
        n = 0.0
        for p in block.parameters():
            if p.grad is not None:
                n += p.grad.detach().float().pow(2).sum().item()
        out.append(math.sqrt(n))
    model.zero_grad(set_to_none=True)
    return out


@dataclass
class TrainResult:
    history: list[dict]
    eval: dict
    weight_norm: float
    rel_weight_change: float
    final_grad_norm: float
    final_lr: float
    elr: float
    router_probe: dict | None = None
    final_state: dict | None = None
    noise_scale: float | None = None


def train_model(
    model: TinyLM,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    mcfg: ModelConfig,
    tcfg: TrainConfig,
    optimizer_name: str = "adamw",
    schedule: str = "cosine",
    seed: int = 0,
    micro_batch: int | None = None,
    decay_start: int | None = None,
    probe_steps: list[int] | None = None,
    probe_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    noise_probe_step: int | None = None,
    noise_micro_batch: int = 8,
) -> TrainResult:
    set_seed(seed)
    model.train()
    initial = {n: p.detach().clone() for n, p in matrix_params(model)}
    opt = make_optimizer(optimizer_name, model.parameters(), tcfg.lr, tcfg.weight_decay)
    g = torch.Generator().manual_seed(seed + 999)
    accum = max(1, tcfg.batch_size // (micro_batch or tcfg.batch_size))
    history = []
    router_probe = None
    noise_scale = None
    for step in range(tcfg.steps):
        lr = lr_at_tail(step, tcfg, decay_start) if decay_start is not None else lr_at(step, tcfg, schedule)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            x, y = sample_batch(train_ids, micro_batch or tcfg.batch_size, mcfg.ctx, g)
            hidden, aux = model(x)
            loss, _ = model.mtp_loss(hidden, y, aux)
            (loss / accum).backward()
            total += loss.item() / accum
        gn = grad_norm(model)
        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            history.append({
                "step": step,
                "loss": total,
                "lr": lr,
                "grad_norm": gn,
                "param_norm": param_norm(model),
            })
        if probe_steps and probe_batches and step in probe_steps:
            router_probe = collect_expert_load(model, probe_batches)
        if noise_probe_step is not None and step == noise_probe_step:
            noise_scale = grad_noise_scale(model, train_ids, mcfg.ctx, noise_micro_batch, n_micro=12, seed=seed + 555)
    ev = evaluate(model, val_ids, tcfg.batch_size, mcfg.ctx, tcfg.eval_batches)
    wn = param_norm(model)
    final_grad = history[-1]["grad_norm"] if history else float("nan")
    nonzero_lr = [h for h in history if h["lr"] > 0]
    elr_point = nonzero_lr[-1] if nonzero_lr else (history[-1] if history else None)
    return TrainResult(
        history=history,
        eval=ev,
        weight_norm=wn,
        rel_weight_change=relative_weight_change(model, initial),
        final_grad_norm=final_grad,
        final_lr=history[-1]["lr"] if history else float("nan"),
        elr=(elr_point["lr"] * elr_point["grad_norm"] / (elr_point["param_norm"] + 1e-12)) if elr_point else float("nan"),
        router_probe=router_probe,
        final_state={"state_dict": copy.deepcopy(model.state_dict())},
        noise_scale=noise_scale,
    )


def grad_noise_scale(model: TinyLM, ids: torch.Tensor, ctx: int, micro_batch: int,
                     n_micro: int = 16, seed: int = 777) -> float:
    """||Var(g)|| / ||E[g]||^2 over micro-batches (gradient noise scale proxy)."""
    g = torch.Generator().manual_seed(seed)
    grads = []
    for _ in range(n_micro):
        x, y = sample_batch(ids, micro_batch, ctx, g)
        model.zero_grad(set_to_none=True)
        hidden, aux = model(x)
        loss, _ = model.mtp_loss(hidden, y, aux)
        loss.backward()
        grads.append(torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]).detach())
    G = torch.stack(grads)
    mean = G.mean(dim=0)
    var = G.var(dim=0, unbiased=True).sum()
    return float(var / (mean.pow(2).sum() + 1e-12))
