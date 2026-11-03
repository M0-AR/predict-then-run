from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptr.ledger import verify  # noqa: E402


def mean(block: dict, key: str) -> float:
    return float(block[key]["mean"])


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}".replace("-", "\u2212")


def plain(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def load_data() -> dict:
    scores = json.loads((ROOT / "results/scores.json").read_text())
    predictions = yaml.safe_load((ROOT / "preregistry/predictions.yaml").read_text())
    by_id = {p["id"]: p for p in predictions}
    raw = {
        name: json.loads((ROOT / f"results/raw/{name}.json").read_text())
        for name in ("e1", "e2", "e3", "e4", "e5", "e6", "market")
    }
    manifest = json.loads((ROOT / "results/raw/run_manifest.json").read_text())
    ledger = verify(ROOT / "preregistry/predictions.yaml", ROOT / "preregistry/ledger.jsonl")
    agg = scores["aggregate"]

    rows = []
    for score in scores["scores"]:
        pred = by_id[score["id"]]
        predicted = pred.get("predicted_relative_effect")
        if predicted is None:
            predicted_label = f"metric {pred['comparator']} {pred['threshold']:g}"
            predicted_value = None
        else:
            predicted_label = signed(float(predicted))
            predicted_value = float(predicted)
        rows.append(
            {
                "id": score["id"],
                "experiment": score["experiment"],
                "claim": score["claim"],
                "kind": score["kind"],
                "confidence": score["confidence"],
                "predicted_label": predicted_label,
                "predicted_value": predicted_value,
                "predicted_interval": pred.get("predicted_interval"),
                "falsifier": pred.get("falsifier"),
                "citation": pred.get("citation"),
                "measured": score["measured"],
                "ci_low": score["ci_low"],
                "ci_high": score["ci_high"],
                "verdict": score["verdict"],
                "sign_match": score["sign_match"],
                "interval_covered": score["interval_covered"],
                "absolute_error": score["absolute_error"],
            }
        )

    e1 = raw["e1"]["by_condition"]
    e2 = raw["e2"]["by_condition"]
    e3 = raw["e3"]["by_condition"]
    e4 = raw["e4"]["by_condition"]
    e5 = raw["e5"]["by_condition"]
    e6 = raw["e6"]["by_condition"]
    market = raw["market"]
    wf = market["walk_forward"]
    shuffle = market["shuffled_control"]
    vol = market["volatility"]

    groups = {
        "e1": ("E1", "Weight decay"),
        "e2": ("E2", "Muon vs AdamW"),
        "e3": ("E3", "Multi-token prediction"),
        "e4": ("E4", "MoE routing"),
        "e5": ("E5", "Learning-rate tail"),
        "e6": ("E6", "Learned memory gating"),
        "market": ("M", "Live-market probe"),
    }
    experiments = [
        {
            "id": "e1",
            "code": "E1",
            "name": "Weight decay",
            "question": "Does weight decay do at tiny scale what the literature says?",
            "headline": f"Weight norm {plain(mean(e1['0.0'], 'weight_norm'))} \u2192 {plain(mean(e1['0.1'], 'weight_norm'))}",
            "detail": f"Validation loss {plain(mean(e1['0.0'], 'val_loss'))} \u2192 {plain(mean(e1['0.1'], 'val_loss'))}. The direction was right; the size was smaller than predicted.",
            "seconds": raw["e1"]["_seconds"],
        },
        {
            "id": "e2",
            "code": "E2",
            "name": "Muon vs AdamW",
            "question": "Does Muon beat AdamW under a matched tuning budget?",
            "headline": f"Best validation loss {plain(mean(e2['128_muon'], 'val_loss'))} vs {plain(mean(e2['128_adamw'], 'val_loss'))}",
            "detail": "At width 128, Muon is 28.3% lower than AdamW. At width 64: 1.597 vs 2.057. The advantage is larger than predicted.",
            "seconds": raw["e2"]["_seconds"],
        },
        {
            "id": "e3",
            "code": "E3",
            "name": "Multi-token prediction",
            "question": "Does an extra prediction head help or interfere with the main head?",
            "headline": f"Main-head loss {plain(mean(e3['baseline'], 'val_loss_h0'))} \u2192 {plain(mean(e3['mtp_downweighted'], 'val_loss_h0'))}",
            "detail": f"The MTP head's own loss is {plain(mean(e3['mtp_downweighted'], 'val_loss_h1'))}. The predicted gradient alignment failed badly.",
            "seconds": raw["e3"]["_seconds"],
        },
        {
            "id": "e4",
            "code": "E4",
            "name": "MoE routing",
            "question": "Does sparse routing collapse, persist, or stay balanced?",
            "headline": f"Router max share {plain(mean(e4['0.0'], 'max_share'))} \u2192 {plain(mean(e4['0.01'], 'max_share'))}",
            "detail": f"Entropy {plain(mean(e4['0.0'], 'entropy_norm'))} \u2192 {plain(mean(e4['0.01'], 'entropy_norm'))}; validation loss {plain(mean(e4['0.0'], 'val_loss'))} \u2192 {plain(mean(e4['0.01'], 'val_loss'))}.",
            "seconds": raw["e4"]["_seconds"],
        },
        {
            "id": "e5",
            "code": "E5",
            "name": "Learning-rate tail",
            "question": "Does a cosine tail pay off more when gradient noise is high?",
            "headline": f"Cosine tail {plain(mean(e5['16_constant'], 'val_loss'))} \u2192 {plain(mean(e5['16_cosine_tail'], 'val_loss'))}",
            "detail": "Benefit +0.016 at width 16, \u22120.005 at 32, +0.006 at 8. The noise interaction was not supported.",
            "seconds": raw["e5"]["_seconds"],
        },
        {
            "id": "e6",
            "code": "E6",
            "name": "Learned memory gating",
            "question": "Can a model learn when to trust a stale external memory?",
            "headline": f"NEW-fact top-1 {plain(mean(e6['always_hints'], 'new_canon_on_top1'))} \u2192 {plain(mean(e6['gated_hints'], 'new_canon_on_top1'))}",
            "detail": "OLD-fact top-1 falls from 1.000 to 0.083. The gate recovers updated facts by suppressing hints, not by clean selective trust.",
            "seconds": raw["e6"]["_seconds"],
        },
        {
            "id": "market",
            "code": "M",
            "name": "Live-market probe",
            "question": "Does a walk-forward linear model find an edge after costs?",
            "headline": f"Directional accuracy {pct(wf['directional_accuracy'])}",
            "detail": f"MSE skill {pct(wf['mse_skill'])}; Sharpe after cost {plain(wf['sharpe_after_cost'])}; volatility ACF(1) {plain(vol['acf']['1'])}. No tradable linear edge.",
            "seconds": raw["market"]["_seconds"],
        },
    ]
    for exp in experiments:
        prefix = exp["id"]
        selected = [r for r in rows if r["experiment"].startswith(prefix)]
        exp["counts"] = {
            "total": len(selected),
            "confirmed": sum(r["verdict"] == "CONFIRMED" for r in selected),
            "refuted": sum(r["verdict"] == "REFUTED" for r in selected),
            "inconclusive": sum(r["verdict"] == "INCONCLUSIVE" for r in selected),
        }

    data = {
        "generated": scores["scored_utc"],
        "aggregate": agg,
        "ledger": {
            "entries": ledger["entries"],
            "head": ledger["head"],
            "valid": ledger["valid"],
        },
        "manifest": {
            "seconds": manifest["seconds"],
            "minutes": round(manifest["seconds"] / 60, 1),
            "python": manifest["python"],
            "torch": manifest["torch"],
            "seeds": manifest["seeds"],
            "steps": manifest["steps"],
            "finished_utc": manifest["finished_utc"],
        },
        "calibration": scores["calibration"],
        "predictions": rows,
        "experiments": experiments,
        "market": {
            "symbol": market["meta"]["symbol"],
            "first_date": market["meta"]["first"],
            "last_date": market["meta"]["last"],
            "n_rows": market["n_rows"],
            "n_folds": wf["n_folds"],
            "n_trades": wf["n_trades"],
            "directional_accuracy": wf["directional_accuracy"],
            "mse_skill": wf["mse_skill"],
            "sharpe_after_cost": wf["sharpe_after_cost"],
            "vol_acf1": vol["acf"]["1"],
            "shuffled_accuracy": shuffle["directional_accuracy"],
        },
        "groups": groups,
    }
    return data


def main() -> None:
    data = load_data()
    template = (ROOT / "site/template.html").read_text()
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    agg = data["aggregate"]
    replacements = {
        "{{N_PREDICTIONS}}": str(agg["n_predictions"]),
        "{{N_DECIDED}}": str(agg["n_decided"]),
        "{{N_CONFIRMED}}": str(agg["n_confirmed"]),
        "{{N_REFUTED}}": str(agg["n_refuted"]),
        "{{N_INCONCLUSIVE}}": str(agg["n_inconclusive"]),
        "{{BRIER_SKILL}}": f"{agg['brier_skill_vs_half']:.3f}",
        "{{COVERAGE}}": f"{agg['interval_coverage'] * 100:.1f}%",
        "{{RUNTIME_MIN}}": f"{data['manifest']['minutes']:.1f}",
        "{{LEDGER_HEAD_SHORT}}": data["ledger"]["head"][:12] + "\u2026",
        "{{N_TESTS}}": "15",
        "{{GENERATED}}": data["generated"],
    }
    html = template.replace("__PTR_DATA__", payload)
    for key, value in replacements.items():
        html = html.replace(key, value)
    (ROOT / "preview.html").write_text(html)
    (ROOT / "assets/data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    (ROOT / "index.html").write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta http-equiv="refresh" content="0; url=preview.html">\n'
        '<link rel="canonical" href="preview.html">\n'
        "<title>Predict, Then Run</title></head>\n"
        '<body style="font:16px system-ui;padding:2rem;background:#0b1220;color:#f8fafc">\n'
        '<p>Opening <a style="color:#5eead4" href="preview.html">the interactive preview</a>\u2026</p>\n'
        "</body></html>\n"
    )
    print(f"wrote preview.html, index.html, assets/data.json ({len(data['predictions'])} claims)")


if __name__ == "__main__":
    main()
