from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 96
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 96
    d_ff: int = 384
    ctx: int = 96
    dropout: float = 0.0
    mtp_heads: int = 1
    mtp_weight: float = 0.3
    moe_experts: int = 0
    moe_top_k: int = 1
    aux_loss_weight: float = 0.0
    router_init_std: float = 0.02


@dataclass
class TrainConfig:
    batch_size: int = 24
    steps: int = 300
    lr: float = 0.003
    min_lr_ratio: float = 0.0
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_batches: int = 16
    log_every: int = 25


@dataclass
class Config:
    seed: int = 0
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    device: str = "cpu"
    torch_threads: int = 8
    data: dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    market: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    model = ModelConfig(**raw.get("model", {}))
    train = TrainConfig(**raw.get("train", {}))
    data = dict(raw.get("data", {}))
    if "e6" in raw:
        data["e6"] = raw["e6"]
    return Config(
        seed=raw.get("seed", 0),
        seeds=list(raw.get("seeds", [0, 1, 2, 3, 4])),
        device=raw.get("device", "cpu"),
        torch_threads=int(raw.get("torch_threads", 8)),
        data=data,
        model=model,
        train=train,
        market=raw.get("market", {}),
    )
