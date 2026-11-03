from __future__ import annotations

import argparse
import json

from .ledger import freeze, verify
from .predictions import load_predictions


def main() -> int:
    ap = argparse.ArgumentParser(prog="ptr", description="predict-then-run utilities")
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("freeze", help="freeze predictions into the hash-chained ledger")
    f.add_argument("--predictions", default="preregistry/predictions.yaml")
    f.add_argument("--ledger", default="preregistry/ledger.jsonl")

    v = sub.add_parser("verify", help="verify the ledger against the predictions")
    v.add_argument("--predictions", default="preregistry/predictions.yaml")
    v.add_argument("--ledger", default="preregistry/ledger.jsonl")

    c = sub.add_parser("check", help="schema-check the prediction file")
    c.add_argument("--predictions", default="preregistry/predictions.yaml")

    args = ap.parse_args()
    if args.command == "freeze":
        entries = freeze(args.predictions, args.ledger)
        check = verify(args.predictions, args.ledger)
        print(json.dumps({"entries": len(entries), "valid": check["valid"], "head": check["head"]}, indent=2))
        return 0 if check["valid"] else 1
    if args.command == "verify":
        check = verify(args.predictions, args.ledger)
        print(json.dumps(check, indent=2))
        return 0 if check["valid"] else 1
    preds = load_predictions(args.predictions)
    print(json.dumps({"predictions": len(preds), "ids": [p["id"] for p in preds]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
