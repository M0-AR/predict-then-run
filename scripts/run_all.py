#!/usr/bin/env python3
"""Run the full predict-then-run benchmark and persist raw results."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ptr.config import load_config
from ptr.experiments.common import save_result
from ptr.market import run_market_probe

EXPERIMENTS = {
    "e1": "e1_weight_decay",
    "e2": "e2_muon",
    "e3": "e3_mtp",
    "e4": "e4_moe",
    "e5": "e5_lr_decay",
    "e6": "e6_hints",
}


def _summary(name: str, out: dict) -> None:
    print(f"[{name}] done in {out.get('_seconds', 0):.1f}s", flush=True)
    for pid, res in out.get("predictions", {}).items():
        measured = res.get("measured", res.get("measured_relative"))
        lo = res.get("ci_low", res.get("ci_low_relative"))
        hi = res.get("ci_high", res.get("ci_high_relative"))
        print(f"  {pid}: {measured:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiments", default="e1,e2,e3,e4,e5,e6,market")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out", default="results/raw")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seeds:
        cfg.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.steps:
        cfg.train.steps = args.steps
    torch.set_num_threads(cfg.torch_threads)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    chosen = [s.strip() for s in args.experiments.split(",") if s.strip()]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    for name in chosen:
        t = time.time()
        if name == "market":
            out = run_market_probe(cfg.market)
            out["_seconds"] = time.time() - t
            save_result("market", out, results_dir=args.out)
            _summary("market", out)
            continue
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name}")
        if name == "e6":
            cfg.data.setdefault("e6", {})
            cfg.data["e6"]["seeds"] = list(cfg.seeds[:3])
        mod = importlib.import_module(f"ptr.experiments.{EXPERIMENTS[name]}")
        out = mod.run(cfg)
        out["_seconds"] = time.time() - t
        save_result(name, out, results_dir=args.out)
        _summary(name, out)

    manifest = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.time() - t0,
        "experiments": chosen,
        "seeds": list(cfg.seeds),
        "steps": cfg.train.steps,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": cfg.to_dict(),
    }
    Path(args.out, "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"total {manifest['seconds']:.1f}s -> {args.out}/run_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
