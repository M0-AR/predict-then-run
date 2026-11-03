#!/usr/bin/env python3
"""Freeze the pre-registered predictions into a hash-chained ledger."""
from __future__ import annotations

import argparse
import json

from ptr.ledger import freeze, verify


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="preregistry/predictions.yaml")
    ap.add_argument("--ledger", default="preregistry/ledger.jsonl")
    args = ap.parse_args()
    entries = freeze(args.predictions, args.ledger)
    check = verify(args.predictions, args.ledger)
    print(json.dumps({"entries": len(entries), "head": check["head"], "valid": check["valid"]}, indent=2))
    return 0 if check["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
