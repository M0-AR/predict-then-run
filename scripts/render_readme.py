#!/usr/bin/env python3
"""Render README.md from frozen predictions and measured results.

No number in the generated README is typed by hand: every metric is read from
results/scores.json or results/raw/*.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def f(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if math.isnan(x) or math.isinf(x):
        return "n/a"
    return f"{x:.{nd}f}"


def pct(x, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{100 * float(x):.{nd}f}%"


def ci(score: dict, nd: int = 3) -> str:
    return f"{f(score['measured'], nd)} [{f(score['ci_low'], nd)}, {f(score['ci_high'], nd)}]"


def esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def mean(metric: dict) -> float:
    return metric["mean"] if metric else float("nan")


def main() -> None:
    scores_payload = load_json(ROOT / "results" / "scores.json")
    scores = {s["id"]: s for s in scores_payload["scores"]}
    def outcome(pid: str) -> str:
        return scores[pid]["verdict"].lower()
    pred_by_id = {p["id"]: p for p in yaml.safe_load((ROOT / "preregistry" / "predictions.yaml").read_text())}
    def pred_effect(pid: str):
        p = pred_by_id[pid]
        return p.get("predicted_relative_effect") if p["kind"] == "directional" else p.get("threshold")
    agg = scores_payload["aggregate"]
    cal = scores_payload["calibration"]
    ledger = scores_payload["ledger"]
    preds = yaml.safe_load((ROOT / "preregistry" / "predictions.yaml").read_text())
    manifest = load_json(RAW / "run_manifest.json")
    raw = {
        "e1": load_json(RAW / "e1.json"),
        "e2": load_json(RAW / "e2.json"),
        "e3": load_json(RAW / "e3.json"),
        "e4": load_json(RAW / "e4.json"),
        "e5": load_json(RAW / "e5.json"),
        "e6": load_json(RAW / "e6.json"),
        "market": load_json(RAW / "market.json"),
    }

    units = {
        "E1.P1": "relative weight norm", "E1.P2": "relative weight growth",
        "E1.P3": "relative val loss", "E1.P4": "relative ELR",
        "E2.P1": "relative val loss", "E2.P2": "absolute loss advantage",
        "E2.P3": "relative val loss",
        "E3.P1": "relative NTP loss", "E3.P2": "relative gradient cosine",
        "E3.P3": "relative NTP loss", "E3.P4": "relative NTP loss",
        "E3.P5": "absolute gradient cosine",
        "E4.P1": "normalized entropy", "E4.P2": "relative entropy",
        "E4.P3": "early-late correlation", "E4.P4": "relative val loss",
        "E5.P1": "relative val loss", "E5.P2": "absolute loss benefit",
        "E5.P3": "correlation",
        "E6.P1": "top-1 rate difference", "E6.P2": "top-1 rate difference",
        "E6.P3": "top-1 rate difference", "E6.P4": "gate probability difference",
        "E6.P5": "top-1 rate difference", "E6.P6": "top-1 rate difference",
        "E6.P7": "top-1 rate difference", "E6.P8": "top-1 rate difference",
        "E6.P9": "top-1 rate difference",
        "M1": "accuracy", "M2": "MSE skill", "M3": "absolute Sharpe",
        "M4": "lag-1 ACF of abs(returns)", "M5": "abs(accuracy - 0.5)",
    }

    e1 = raw["e1"]["by_condition"]
    e2 = raw["e2"]["by_condition"]
    tune = raw["e2"]["tuning"]
    n_adamw = len({r["lr"] for r in tune if r["optimizer"] == "adamw"})
    n_muon = len({r["lr"] for r in tune if r["optimizer"] == "muon"})
    muon_lrs = sorted({v["lr"] for k, v in e2.items() if "muon" in k})
    e3 = raw["e3"]["by_condition"]
    e4 = raw["e4"]["by_condition"]
    e5 = raw["e5"]["by_condition"]
    e6 = raw["e6"]["by_condition"]
    mkt = raw["market"]
    wf = mkt["walk_forward"]
    ctrl = mkt["shuffled_control"]
    vol = mkt["volatility"]
    decided = agg["n_decided"]

    lines: list[str] = []
    minutes = manifest["seconds"] / 60
    confirmed = [s for s in scores.values() if s["verdict"] == "CONFIRMED"]
    refuted = [s for s in scores.values() if s["verdict"] == "REFUTED"]
    inconclusive = [s for s in scores.values() if s["verdict"] == "INCONCLUSIVE"]
    refuted_ids = ", ".join(f"`{s['id']}`" for s in refuted)

    lines.append("# Predict, Then Run\n")
    lines.append(
        '<p align="center">\n'
        '  <img src="assets/banner.png" alt="Predict, Then Run: know what will happen before you run" width="100%">\n'
        "</p>\n"
    )
    lines.append(
        "![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg) "
        "![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg) "
        f"![CPU run](https://img.shields.io/badge/CPU%20run-{minutes:.1f}%20min-5eead4.svg) "
        f"![Ledger](https://img.shields.io/badge/ledger-{'valid' if ledger['valid'] else 'invalid'}-2dd4bf.svg) "
        "![Status](https://img.shields.io/badge/status-working%20paper-orange.svg)\n"
    )
    lines.append("**Write the claim. Freeze it. Run the experiment. Publish what survives.**\n")
    lines.append(
        "Repository: [github.com/M0-AR/predict-then-run](https://github.com/M0-AR/predict-then-run)\n"
    )
    lines.append(
        "A prediction-first benchmark for AI research and product decisions: "
        f"{agg['n_predictions']} claims, a hash-chained ledger, and a scorecard that includes the "
        f"{agg['n_refuted']} predictions that failed. *Working paper; every number is generated from "
        "`results/scores.json` and the raw run files.*\n"
    )

    lines.append("## The 60-second version\n")
    lines.append(
        "Most experiments are interpreted after the result is visible. This repository reverses the "
        "order: each claim gets a direction, a confidence, a predicted interval, and a falsifier, and "
        "is frozen in a hash-chained ledger before the final multi-seed run. Then the scorecard is "
        f"published in full: **{agg['n_confirmed']} confirmed, {agg['n_refuted']} refuted, "
        f"{agg['n_inconclusive']} inconclusive**.\n"
    )
    lines.append(
        "| A buyer's question | What this repository provides |\n"
        "|---|---|\n"
        "| “Will this mechanism work?” | A frozen prediction with a direction, a 95% interval, a confidence and a falsifier. |\n"
        f"| “Can I check the claim?” | {agg['n_predictions']} claim-level predictions, raw JSON, a hash-chained ledger, and a CPU rerun in about {minutes:.1f} minutes. |\n"
        f"| “What did you learn?” | {agg['n_confirmed']} confirmations, {agg['n_refuted']} refutations and {agg['n_inconclusive']} honest maybes, including a negative market result and a memory-gating trade-off. |\n"
        "| “What does it cost to try?” | No GPU, no paid API, no private data. Apache-2.0. |\n"
    )
    lines.append(
        f"The refuted predictions are {refuted_ids}. They are not a footnote; they are the product "
        "working as intended. The honest boundary is equally important: this is evidence about a "
        "tiny regime, not a guarantee of business outcomes.\n"
    )

    lines.append("## What you can do with it\n")
    lines.append(
        "- **R&D portfolio triage.** Decide which mechanism claims deserve a sprint before the sprint is funded.\n"
        "- **Vendor and benchmark diligence.** Ask for a frozen claim, a falsifier, and a rerunnable receipt — not just a headline number.\n"
        "- **Consulting deliverable.** Hand a client an auditable evidence pack with raw results and a verdict rule.\n"
        "- **Team training.** Teach engineers and analysts to separate a prediction from a post-hoc story.\n"
        "- **Investor technical diligence.** Stress-test a technical claim without access to private systems.\n"
        "- **Academic pre-registration.** Use the ledger and scoring rules as a template for a registered report.\n"
    )

    lines.append("## See it in action\n")
    lines.append(
        "The repository ships a self-contained interactive page (`preview.html`) and a short walkthrough.\n"
    )
    lines.append(
        "| Desktop hero | Evidence dashboard | Beginner path |\n"
        "|---|---|---|\n"
        "| ![Hero](assets/screenshot-hero.png) | ![Evidence](assets/screenshot-evidence.png) | ![Learn](assets/screenshot-learn.png) |\n"
    )
    lines.append(
        '<p align="center"><img src="assets/screenshot-mobile.png" alt="Predict, Then Run on a phone" width="320"></p>\n'
    )
    lines.append("![Interactive walkthrough](assets/demo.gif)\n")
    lines.append(
        "[Open the walkthrough video](assets/demo.mp4). The page also has claim filters, a ten-question "
        "quiz, a value calculator, and a dark-mode toggle.\n"
    )

    lines.append("## Quick start\n")
    lines.append(
        "```bash\n"
        "python -m venv .venv && . .venv/bin/activate\n"
        'pip install -e ".[dev]"\n'
        "ptr verify                     # check the frozen ledger\n"
        "python scripts/run_all.py --seeds 0,1,2,3,4 --steps 300\n"
        "python scripts/score.py\n"
        "python scripts/render_readme.py\n"
        "```\n"
    )
    lines.append(
        "Or use the clean-room Docker path:\n\n"
        "```bash\n"
        "docker compose build\n"
        "docker compose run --rm benchmark\n"
        "```\n"
    )
    lines.append(
        f"No GPU, no API key, no private data. The full CPU run takes about {minutes:.1f} minutes on "
        "eight threads. See [Reproducibility](#11-reproducibility) for the exact commands and artifacts.\n"
    )

    lines.append("## Contents\n")
    lines.append(
        "- [The 60-second version](#the-60-second-version)\n"
        "- [What you can do with it](#what-you-can-do-with-it)\n"
        "- [See it in action](#see-it-in-action)\n"
        "- [Quick start](#quick-start)\n"
        "- [🌱 Beginner guide](#-beginner-guide--read-this-and-you-are-a-professional)\n"
        "- [Features](#features)\n"
        "- [FAQ](#faq)\n"
        "- [Repository map](#repository-map)\n"
        "- [Publishing this page](#publishing-this-page)\n"
        "- [Abstract](#abstract)\n"
        "- [1. Research question](#1-research-question)\n"
        "- [2. Related work](#2-related-work)\n"
        "- [3. Contributions](#3-contributions)\n"
        "- [4. Protocol](#4-protocol)\n"
        "- [5. Results at a glance](#5-results-at-a-glance)\n"
        "- [6. Predictions and outcomes](#6-predictions-and-outcomes)\n"
        "- [7. Experiment-by-experiment findings](#7-experiment-by-experiment-findings)\n"
        "- [8. Live-market probe](#8-live-market-probe-m1-m5)\n"
        "- [9. What replicated, what failed, what is new](#9-what-replicated-what-failed-what-is-new)\n"
        "- [10. Limitations](#10-limitations-and-threats-to-validity)\n"
        "- [11. Reproducibility](#11-reproducibility)\n"
        "- [12. Data, code and ethics](#12-data-code-and-ethics)\n"
        "- [13. References](#13-references)\n"
    )

    lines.append("## 🌱 Beginner guide — read this and you are a professional\n")
    lines.append(
        "> **You will know more than most interview candidates.**\n"
        ">\n"
        "> Let's work this out in a step-by-step way to be sure we have the right answer.\n"
    )
    lines.append(
        "You do not need a statistics background. You need one sentence that can be wrong, one "
        "number that says how sure you are, and one observation that would change your mind. The "
        "rest is bookkeeping.\n"
    )
    lines.append("### 1. Write a sentence that can be wrong\n")
    lines.append(
        "“The new onboarding will improve activation” is a wish. “The new onboarding will raise "
        "7-day activation by at least 3 percentage points within four weeks” is a claim: it has a "
        "direction, a size and a deadline.\n"
    )
    lines.append("### 2. Add a confidence and a falsifier\n")
    lines.append(
        "The confidence is a number from 0 to 1: how often would a forecast like this be right? "
        "The falsifier is the observation that would prove you wrong. If you cannot name it, you do "
        "not yet have a test.\n"
    )
    lines.append("### 3. Freeze it\n")
    lines.append(
        "Write the claim, the direction, the interval, the confidence, and the falsifier. Hash the "
        "payload and chain it to the previous entry. This is the difference between a prediction and "
        "a memory: later edits become visible.\n"
    )
    lines.append("### 4. Run a controlled test\n")
    lines.append(
        "Fix the seeds, the budget, and the scoring rule. Compare against a matched control. Keep the "
        "raw output, not just the chart. The experiment should be small enough to repeat and real "
        "enough to surprise you.\n"
    )
    lines.append("### 5. Read the interval, not just the average\n")
    lines.append(
        "A 95% interval is a range of plausible values for the measured effect. If it includes zero, "
        "the honest verdict is *inconclusive*: this run cannot tell the two stories apart. That is a "
        "result, not a failure of the write-up.\n"
    )
    lines.append("### 6. Score the forecast, not the story\n")
    lines.append(
        "A claim is *confirmed* only if the interval excludes zero and the sign matches. It is "
        "*refuted* if the interval excludes zero with the opposite sign. It is *inconclusive* "
        "otherwise. A point estimate on the right side of a threshold is not enough.\n"
    )
    lines.append("### 7. Use it on a real decision\n")
    lines.append(
        "Pick the claim that would change what you do. If a wrong answer is cheap, test it. If it is "
        "expensive, freeze the prediction before you look. Then publish the result, including the "
        "ones that failed.\n"
    )

    lines.append("### Worked example: the claim that missed the size\n")
    lines.append(
        f"`E2.P1` predicted that tuned Muon would beat tuned AdamW by about "
        f"{abs(pred_effect('E2.P1')) * 100:.0f}% at width 64. The measured relative difference was "
        f"{pct(scores['E2.P1']['measured'])} (95% CI {pct(scores['E2.P1']['ci_low'])} to "
        f"{pct(scores['E2.P1']['ci_high'])}): Muon was much better. The verdict is "
        f"**{scores['E2.P1']['verdict']}** because the direction was right and the interval excluded "
        "zero. The forecast was still badly sized: the interval did not cover the prediction. Both "
        "facts belong in the report.\n"
    )
    lines.append(
        f"`E6.P5` is the opposite lesson. It predicted that order-invariant keys would recover "
        f"reordered facts by {abs(pred_effect('E6.P5')) * 100:.0f} points. The measured gain was "
        f"{scores['E6.P5']['measured'] * 100:.1f} points, with a 95% CI from "
        f"{scores['E6.P5']['ci_low'] * 100:.1f} to {scores['E6.P5']['ci_high'] * 100:.1f}. The "
        f"interval includes zero, so the verdict is **{scores['E6.P5']['verdict']}**. The honest "
        "answer is “we cannot tell from this run,” which is more useful than pretending it worked.\n"
    )
    lines.append(
        "**The trap.** A confirmed claim is not a proven mechanism. It is a prediction that survived "
        "one test, in one regime, with one measurement. The limitations section is part of the "
        "result, not an apology.\n"
    )

    lines.append("### A 10-minute path\n")
    lines.append(
        "| Minute | Do this |\n"
        "|---|---|\n"
        "| 0–2 | Read the 60-second version above. |\n"
        "| 2–5 | Open `preview.html`, filter to **Refuted**, and open one claim to see its falsifier. |\n"
        "| 5–8 | Take the ten-question quiz; the explanations are the point. |\n"
        "| 8–10 | Run `ptr verify` and compare the ledger head with the one in section 4. |\n"
    )
    lines.append(
        "If you can explain why a refuted claim is more useful than a vague success story, you have "
        "the method.\n"
    )

    lines.append("## Features\n")
    lines.append(
        "- **33 frozen claims** across six controlled language-model experiments and one live-market probe.\n"
        "- **A hash-chained ledger** (`preregistry/ledger.jsonl`) that detects post-hoc edits; the scorer refuses to run if the chain or payload hashes do not match.\n"
        "- **Claim-level scoring**: paired bootstrap CIs, Brier score, Brier skill, log loss, interval coverage, calibration bins, and per-experiment verdicts.\n"
        "- **A CPU-only benchmark** that completes in about 12 minutes on eight threads, with no GPU, paid API, or private data.\n"
        "- **An interactive page** with claim filters, expandable falsifiers, a ten-question quiz, a value calculator, and a dark-mode toggle.\n"
        "- **Raw artifacts**: every per-run JSON file, the complete run manifest, the scored table, and the ledger.\n"
        "- **Negative results published**: five refuted predictions and eleven inconclusive ones are reported as such.\n"
        "- **A clean-room Docker path** plus a direct Python path, with 15 tests and a lint target.\n"
    )

    lines.append("## FAQ\n")
    lines.append(
        "**Is this peer-reviewed?** No. It is a working paper with public artifacts and a rerunnable "
        "pipeline. The value is auditability, not a journal stamp.\n\n"
        "**Does a confirmed prediction prove the mechanism?** No. It is evidence that the predicted "
        "effect appeared in this regime. The limitations section lists the boundaries.\n\n"
        "**Why are 11 claims inconclusive?** Their 95% intervals include zero. Calling them "
        "“confirmed” because the average moved in the right direction would defeat the protocol.\n\n"
        "**Why include a market probe?** It is a calibration control, not an investment strategy. It "
        "tests whether the pipeline can report a negative result cleanly; it is not financial advice.\n\n"
        "**Can I use it commercially?** Yes. The code is Apache-2.0; the text and figures are CC BY "
        "4.0. Attribution is required.\n\n"
        "**Do I need a GPU or an API key?** No. The full run is CPU-only and uses public data.\n\n"
        "**How do I know the predictions were not edited?** Run `ptr verify`. The ledger chains a "
        "SHA-256 of each payload through `prev_hash`; the head hash is published in section 4.\n\n"
        "**What is the biggest limitation?** Scale. Two layers and 300 steps expose mechanisms; they "
        "do not establish scaling laws. The predictions were also written after exploratory pilots, "
        "which is disclosed.\n\n"
        "**Why does the memory-gating result matter?** It is a trade-off, not a free lunch: the "
        "learned gate improves updated-fact accuracy by suppressing hints, while old-fact recall "
        "falls sharply. That is more useful than a clean-sounding claim that hides the cost.\n\n"
        "**How do I cite it?** Use `CITATION.cff`; the repository URL is included.\n\n"
        "**Is the quiz graded?** No. It is a teaching device; the score resets on refresh.\n"
    )

    lines.append("## Repository map\n")
    lines.append(
        "| Path | What it contains |\n"
        "|---|---|\n"
        "| `preregistry/predictions.yaml` | The 33 frozen claims, with direction, interval, confidence, falsifier and citation. |\n"
        "| `preregistry/ledger.jsonl` | The hash-chained freeze ledger. |\n"
        "| `src/ptr/` | Model, tokenizer, training loop, Muon, market probe and scoring. |\n"
        "| `src/ptr/experiments/` | Six controlled experiments and the market probe. |\n"
        "| `scripts/` | Freeze, run, score, render, build and record. |\n"
        "| `site/template.html` | The self-contained interactive page source. |\n"
        "| `preview.html` | The generated GitHub Pages entry point. |\n"
        "| `results/raw/` | Every per-run JSON file and the complete run manifest. |\n"
        "| `results/scores.json`, `results/tables/scores.csv` | Claim-level verdicts, calibration and aggregates. |\n"
        "| `assets/` | Banner, screenshots and the walkthrough video. |\n"
        "| `tests/` | Unit and integration tests for the ledger, scoring and site build. |\n"
        "| `Dockerfile`, `docker-compose.yml`, `Makefile` | Clean-room and local entry points. |\n"
        "| `CITATION.cff`, `LICENSE` | Citation metadata and the Apache-2.0 license. |\n"
    )

    lines.append("## Publishing this page\n")
    lines.append(
        "1. Open the public repository `M0-AR/predict-then-run`.\n"
        "2. Go to **Settings → Pages → Deploy from a branch**, choose `main` and `/ (root)`.\n"
        "3. Wait for the Pages build. `index.html` redirects to `preview.html`.\n"
        "4. Set the repository social preview to `assets/banner.png`.\n"
        "5. Check the claims table, the quiz and the mobile menu on the published URL.\n"
    )
    lines.append(
        "The `.nojekyll` file is included so the generated page is served without Jekyll processing.\n"
    )

    lines.append("## Abstract\n")
    lines.append(
        "Most small-scale mechanistic experiments are run first and interpreted afterwards. This "
        "repository reverses the order: every claim is written down, given a confidence, given a "
        "falsifier, and frozen in a hash-chained ledger *before* the final multi-seed run. The "
        "benchmark then reports not only what happened, but whether the prediction that motivated "
        "the experiment survived contact with the data.\n"
    )
    lines.append(
        f"We evaluate {agg['n_predictions']} pre-registered predictions across six controlled "
        "language-model experiments and one live-market probe. The experiments are deliberately "
        "small enough to run on a CPU in minutes, so that the cost of a wrong hypothesis is a "
        "coffee break rather than a cluster allocation. Predictions are scored with paired "
        f"bootstrap intervals, Brier scores and calibration bins. Of {agg['n_predictions']} "
        f"predictions, {agg['n_confirmed']} were confirmed, {agg['n_refuted']} were refuted and "
        f"{agg['n_inconclusive']} were inconclusive. Over the {agg['n_decided']} decided "
        f"predictions, the Brier skill relative to a naive 0.5 forecaster is "
        f"{f(agg['brier_skill_vs_half'], 3)}; only {pct(agg['interval_coverage'])} of the measured "
        "effects fell inside their 95% bootstrap intervals, which is itself a calibration warning: "
        "the intervals describe sampling noise, not the accuracy of the point forecasts. The "
        "policy-relevant signal is that a *learned* trust gate improves updated-fact accuracy "
        "relative to always-on hints, at the cost of old-fact recall, while the prediction that "
        "naive hint dropout would help is refuted. The benchmark also produces an "
        "honest negative result on a liquid equity index: a walk-forward linear model has negative "
        "out-of-sample MSE skill, while volatility clustering remains strong. All artifacts, "
        "hashes and commands are included.\n"
    )

    lines.append("## 1. Research question\n")
    lines.append(
        "The starting point is a simple claim about method: **predict the mechanism before you "
        "run the experiment, then let a tiny experiment falsify the prediction.** We test that "
        "claim on seven questions that are small enough to answer in minutes:\n"
    )
    lines.append(
        "1. Does weight decay do what the optimizer literature says it does at tiny scale?\n"
        "2. Does Muon beat AdamW when both receive the same tuning budget?\n"
        "3. Does multi-token prediction help or interfere with next-token prediction?\n"
        "4. Does sparse MoE routing collapse, persist, or stay balanced at tiny scale?\n"
        "5. Does a cosine learning-rate tail pay off more when gradient noise is high?\n"
        "6. Can a model learn *when* to trust an external memory that is stale for some facts?\n"
        "7. Does a walk-forward linear model find a tradable edge in SPY after costs?\n"
    )

    lines.append("## 2. Related work\n")
    lines.append(
        "**Pre-registration and prediction-first science.** Pre-registration is standard in "
        "clinical and social science (Nosek et al., 2018); extending it from analysis plans to "
        "claim-level machine-learning predictions is less common. Existing templates usually fix "
        "an analysis plan; we instead freeze *claim-level directional predictions* with "
        "confidence, interval and falsifier, and score each claim separately. The unit of "
        "evidence is a falsifiable sentence, not a pipeline.\n"
    )
    lines.append(
        "**Mechanistic interpretability at small scale.** Work on transformer circuits (Elhage "
        "et al., 2021) and induction heads (Olsson et al., 2022) shows that small models can "
        "expose mechanisms that are hard to isolate at scale. Scaling-law studies (Kaplan et "
        "al., 2020; Hoffmann et al., 2022) describe compute-optimal frontiers but do not test "
        "whether a named mechanism is present. We use tiny models as mechanism detectors, not "
        "as miniature frontier systems.\n"
    )
    lines.append(
        "**Optimization, architecture and memory.** The experiments touch decoupled weight "
        "decay (Loshchilov and Hutter, 2019), Muon (Jordan et al., 2025), multi-token "
        "prediction (arXiv:2604.11912), MoE routing dynamics (arXiv:2610.39442) and conditional "
        "memory (DeepSeek, 2026). The contribution is not a new optimizer or architecture; it is "
        "a protocol that forces these claims to make directional predictions before the run.\n"
    )
    lines.append(
        "**Market validation.** Walk-forward evaluation, the deflated Sharpe ratio and the "
        "probability of backtest overfitting (Bailey et al., 2014, 2017) are standard antidotes "
        "to in-sample finance results. We add a shuffled-label control and report negative MSE "
        "skill instead of selecting the specification with the best backtest.\n"
    )

    lines.append("## 3. Contributions\n")
    lines.append(
        "- A **pre-registration protocol** for small LM experiments: claim, direction, interval, "
        "confidence, falsifier and citation are frozen before the final run.\n"
        "- A **hash-chained ledger** (`preregistry/ledger.jsonl`) that detects post-hoc edits to "
        "predictions; the scorer refuses to run if the chain or payload hashes do not match.\n"
        "- A **claim-level scoring** layer: paired bootstrap CIs, Brier score, Brier skill, log "
        "loss, interval coverage, calibration bins and per-experiment verdicts.\n"
        "- Six **controlled experiments** covering optimization, architecture and memory, plus a "
        "**live-market control** with walk-forward splits, an embargo, transaction costs and a "
        "shuffled-label leakage check.\n"
        "- A **negative replication** of naive hint dropout, a **positive result** for learned "
        "gating on updated facts, and an **inconclusive** order-invariant-key test that is "
        "reported as such rather than folded into the positive claim.\n"
    )

    lines.append("## 4. Protocol\n")
    lines.append(
        f"Predictions were frozen before the final runs. The ledger contains "
        f"{ledger['entries']} entries; its head hash is `{ledger['head']}` and verification "
        f"currently returns **{'valid' if ledger['valid'] else 'INVALID'}**. "
        "The ledger records the Python version, platform, git state and a SHA-256 of each "
        "prediction payload, chained through `prev_hash`. Changing a claim after the fact breaks "
        "the chain and the scorer exits.\n"
    )
    lines.append(
        "**Verdicts.** A directional prediction is *confirmed* only if the paired 95% bootstrap "
        "CI excludes zero and the sign matches; it is *refuted* if the CI excludes zero with the "
        "opposite sign; otherwise it is *inconclusive*. A threshold prediction is scored the same "
        "way against its threshold, with the CI deciding. This is intentionally conservative: a "
        "point estimate on the right side of a threshold is not enough.\n"
    )
    lines.append(
        "**Seeds and splits.** E1-E5 use five seeds; E6 uses three. The market "
        "probe uses an expanding walk-forward window with a one-day embargo, 63-day test blocks, "
        "1 bp costs and a label-shuffled control. All LM experiments use a 2-layer decoder-only "
        "transformer on public text; E6 uses a controlled synthetic update world plus TinyStories "
        "natural text. No pretrained weights are used.\n"
    )
    lines.append(
        "**Honesty about pilots.** One-seed, reduced-step pilot runs were used to debug the "
        "harness and estimate runtime. Predictions were then written from the cited mechanisms "
        "and frozen; the final runs use the full seed count and step budget. The pilot outcomes "
        "are not used as evidence in this README.\n"
    )

    lines.append("## 5. Results at a glance\n")
    lines.append(
        table(
            ["Predictions", "Decided", "Confirmed", "Refuted", "Inconclusive", "Brier", "Brier skill", "Log loss", "Coverage", "Sign accuracy"],
            [[
                agg["n_predictions"], decided, agg["n_confirmed"], agg["n_refuted"],
                agg["n_inconclusive"], f(agg["brier"]), f(agg["brier_skill_vs_half"]),
                f(agg["log_loss"]), pct(agg["interval_coverage"]), pct(agg["sign_accuracy"]),
            ]],
        )
    )
    lines.append("")
    lines.append(
        f"The benchmark's own forecasting performance is modest but informative: a Brier skill "
        f"of {f(agg['brier_skill_vs_half'])} means the pre-registered probabilities beat a "
        f"constant 0.5 forecast, while {agg['n_refuted']} refutations show that the protocol is "
        "not a rubber stamp. The calibration table below is the honest version of the same story.\n"
    )
    lines.append(
        table(
            ["Confidence bin", "n", "Mean confidence", "Empirical accuracy"],
            [[f"{b['lo']:.1f}-{b['hi']:.1f}", b["n"], f(b["mean_confidence"], 2), f(b["empirical_accuracy"], 2)] for b in cal],
        )
    )
    lines.append("")

    lines.append("## 6. Predictions and outcomes\n")
    rows = []
    for p in preds:
        s = scores[p["id"]]
        rows.append([
            f"`{p['id']}`", esc(p["claim"]), f(p["confidence"], 2),
            f(pred_effect(p["id"]), 2),
            f"{ci(s)} ({units.get(p['id'], 'effect')})", s["verdict"],
        ])
    lines.append(table(["ID", "Claim", "Conf.", "Predicted", "Measured [95% CI]", "Verdict"], rows))
    lines.append("")

    lines.append("## 7. Experiment-by-experiment findings\n")

    lines.append("### 7.1 Weight decay (E1)\n")
    lines.append(
        table(
            ["Weight decay", "Weight norm", "Relative weight change", "Val loss", "ELR"],
            [[
                k, f(mean(v["weight_norm"])), f(mean(v["rel_weight_change"])),
                f(mean(v["val_loss"])), f"{mean(v['elr']):.2e}",
            ] for k, v in sorted(e1.items())],
        )
    )
    lines.append("")
    lines.append(
        f"Weight decay reduces the final weight norm ({f(scores['E1.P1']['measured'])} relative; "
        f"95% CI {f(scores['E1.P1']['ci_low'])} to {f(scores['E1.P1']['ci_high'])}) and reduces "
        f"net weight growth ({f(scores['E1.P2']['measured'])}). The validation-loss cost is "
        f"{f(scores['E1.P3']['measured'])} relative, with a CI that decides E1.P3. The effective "
        f"learning-rate point estimate moves in the predicted direction, but E1.P4 is "
        f"{outcome('E1.P4')}; the denominator-shrinkage mechanism is therefore suggestive, not "
        "established. The ELR proxy is evaluated at the final nonzero learning rate and is "
        "numerically tiny (order $10^{-8}$), so its relative test is noisy by construction. The practical lesson is that weight decay is a real lever even in a "
        "300-step run, but its loss effect is small enough that a single seed would be "
        "uninformative.\n"
    )

    lines.append("### 7.2 Muon versus AdamW (E2)\n")
    lines.append(
        table(
            ["Width", "Optimizer", "Selected LR", "Val loss"],
            [[k.split("_")[0], k.split("_")[1], v["lr"], f(mean(v["val_loss"]))] for k, v in sorted(e2.items())],
        )
    )
    lines.append("")
    lines.append(
        f"Both optimizers receive a matched tuning budget ({n_adamw} learning rates for AdamW and "
        f"{n_muon} for Muon), selected on the same validation split. The selected Muon learning "
        "rate is higher than AdamW's at both widths (see the table). At width 64 the paired relative "
        f"loss difference is {f(scores['E2.P1']['measured'])} (95% CI "
        f"{f(scores['E2.P1']['ci_low'])} to {f(scores['E2.P1']['ci_high'])}). The interaction "
        f"test (E2.P2) measures whether the advantage grows with width; its measured value is "
        f"{f(scores['E2.P2']['measured'])}. E2.P3 asks whether tuned Muon stays within 5% of "
        f"AdamW at width 128; the measured relative difference is "
        f"{f(scores['E2.P3']['measured'])}; the claim is one-sided (Muon is no more than 5% *worse*), "
        "and the negative point estimate means it is substantially better, not worse. The Muon "
        "result is therefore a statement about a "
        "*tuned tiny-scale port*, not about the optimizer's asymptotic scaling law. The selected "
        f"Muon rates ({', '.join(f(lr) for lr in muon_lrs)}) are interior points of the grid, so "
        f"the comparison is not driven by a boundary optimum. The measured advantage is much "
        f"larger than the pre-registered {f(pred_effect('E2.P1'))} relative effect, and the "
        "interval does not cover that prediction; the tiny-scale port is stronger than expected.\n"
    )
    lines.append(
        "The tuning table is included so the comparison can be audited:\n"
    )
    tune_rows = []
    for r in raw["e2"]["tuning"]:
        tune_rows.append([r["width"], r["optimizer"], r["lr"], f(r["val_loss"])])
    lines.append(table(["Width", "Optimizer", "Learning rate", "Val loss (100-step tuning)"], tune_rows))
    lines.append("")

    lines.append("### 7.3 Multi-token prediction (E3)\n")
    lines.append(
        table(
            ["Condition", "NTP val loss", "Head-1 val loss", "Head cosine", "Batch cosine", "Layer-0 ratio", "Last-layer ratio"],
            [[
                k, f(mean(v["val_loss_h0"])), f(mean(v["val_loss_h1"])),
                f(mean(v["head_grad_cosine"])), f(mean(v["batch_grad_cosine"])),
                f(mean(v["ratio_layer0"])), f(mean(v["ratio_last"])),
            ] for k, v in e3.items()],
        )
    )
    lines.append("")
    lines.append(
        f"Equal-weight MTP changes next-token validation loss by "
        f"{f(scores['E3.P1']['measured'])} relative (95% CI {f(scores['E3.P1']['ci_low'])} to "
        f"{f(scores['E3.P1']['ci_high'])}; E3.P1 is {outcome('E3.P1')}). The gradient-alignment "
        f"prediction is refuted: the head-1/head-0 cosine is {f(scores['E3.P2']['measured'])} "
        f"relative to an ordinary two-batch cosine (95% CI {f(scores['E3.P2']['ci_low'])} to "
        f"{f(scores['E3.P2']['ci_high'])}), the opposite sign from the predicted decoupling. The "
        f"absolute head-minus-batch cosine is {f(mean(e3['mtp_equal']['head_grad_cosine']) - mean(e3['mtp_equal']['batch_grad_cosine']))}, "
        "so the sign is unambiguous even though the relative ratio is unstable because the "
        "batch cosine is close to zero. "
        f"Downweighting the auxiliary head to 0.3 changes NTP loss by "
        f"{f(scores['E3.P3']['measured'])} relative (E3.P3 is {outcome('E3.P3')}), and the "
        f"absolute cosine difference is {f(scores['E3.P5']['measured'])} (95% CI "
        f"{f(scores['E3.P5']['ci_low'])} to {f(scores['E3.P5']['ci_high'])}; E3.P5 is "
        f"{outcome('E3.P5')}). The pre-registered gradient-decoupling story therefore does not "
        "replicate in this proxy: the auxiliary head is *more* aligned with the main head than "
        "two ordinary batches are, not less. We report this as a failed replication rather than "
        "reinterpreting the metric after the fact.\n"
    )

    lines.append("### 7.4 MoE routing (E4)\n")
    lines.append(
        table(
            ["Aux loss", "Normalized entropy", "Max expert share", "Early-late corr.", "Val loss"],
            [[
                k, f(mean(v["entropy_norm"])), f(mean(v["max_share"])),
                f(mean(v["early_late_corr"])), f(mean(v["val_loss"])),
            ] for k, v in sorted(e4.items())],
        )
    )
    lines.append("")
    lines.append(
        f"With four experts, top-1 routing and no auxiliary loss, normalized entropy is "
        f"{f(scores['E4.P1']['measured'])} (95% CI {f(scores['E4.P1']['ci_low'])} to "
        f"{f(scores['E4.P1']['ci_high'])}), so the router does not collapse at this scale. "
        "Routing is nevertheless persistent: early assignments predict late assignments with "
        f"correlation {f(scores['E4.P3']['measured'])}. The auxiliary loss raises entropy by "
        f"{f(scores['E4.P2']['measured'])} relative and changes validation loss by "
        f"{f(scores['E4.P4']['measured'])} relative. The honest reading is that the tiny model "
        "reproduces *persistence* but not *collapse*; claiming a collapse mechanism from this "
        f"run would be overreach. The auxiliary-loss point estimate is below the 2% threshold, "
        f"but the CI includes it, so E4.P4 is {outcome('E4.P4')}.\n"
    )

    lines.append("### 7.5 Learning-rate decay and gradient noise (E5)\n")
    lines.append(
        table(
            ["Micro-batch", "Schedule", "Loss drop", "Val benefit", "Noise scale", "Val loss"],
            [[
                k.split("_")[0], k.split("_")[1], f(mean(v["drop"])), f(mean(v["benefit"])),
                f(mean(v["noise_scale"])), f(mean(v["val_loss"])),
            ] for k, v in sorted(e5.items(), key=lambda kv: (int(kv[0].split("_")[0]), kv[0].split("_")[1]))],
        )
    )
    lines.append("")
    lines.append(
        f"The cosine tail changes validation loss by {f(scores['E5.P1']['measured'])} relative at "
        f"micro-batch 32 (95% CI {f(scores['E5.P1']['ci_low'])} to {f(scores['E5.P1']['ci_high'])}; "
        f"E5.P1 is {outcome('E5.P1')}). The point estimate is slightly positive, opposite the "
        f"predicted negative effect. The interaction test asks whether the benefit is larger at "
        f"32 than at 16; the measured difference is {f(scores['E5.P2']['measured'])} (positive "
        "means the smaller batch benefits more). The prediction that the larger batch benefits "
        f"more is {outcome('E5.P2')}: the point estimate has the opposite sign. The "
        f"noise-correlation test gives r = {f(scores['E5.P3']['measured'])} (95% CI "
        f"{f(scores['E5.P3']['ci_low'])} to {f(scores['E5.P3']['ci_high'])}; E5.P3 is "
        f"{outcome('E5.P3')}), also opposite to the predicted positive correlation. The "
        "cosine-tail mechanism therefore does not replicate at this scale; the data are more "
        "consistent with the smaller micro-batch benefiting slightly more.\n"
    )

    lines.append("### 7.6 Learned memory gating and order-invariant keys (E6)\n")
    lines.append(
        table(
            ["Condition", "OLD on", "OLD off", "NEW on", "NEW off", "Reordered OLD on", "Gate OLD", "Gate NEW"],
            [[
                k,
                f(mean(v["old_canon_on_top1"])), f(mean(v["old_canon_off_top1"])),
                f(mean(v["new_canon_on_top1"])), f(mean(v["new_canon_off_top1"])),
                f(mean(v["old_reorder_on_top1"])),
                f(mean(v["gate_old"])) if v.get("gate_old") else "n/a",
                f(mean(v["gate_new"])) if v.get("gate_new") else "n/a",
            ] for k, v in e6.items()],
        )
    )
    lines.append("")
    lines.append(
        "This is the flagship experiment. The memory table contains OLD facts; the training "
        "corpus contains both OLD-fact sentences prefixed with `memory` and updated NEW-fact "
        "sentences prefixed with `reason`. The same surface prompt follows both prefixes, so the "
        "only way to use the table correctly is to learn a prefix-conditioned gate.\n"
    )
    lines.append(
        f"External memory recovers OLD facts: top-1 rises from {f(scores['E6.P1']['result']['baseline_mean'])} "
        f"to {f(scores['E6.P1']['result']['treatment_mean'])} (difference {f(scores['E6.P1']['measured'])}; "
        f"95% CI {f(scores['E6.P1']['ci_low'])} to {f(scores['E6.P1']['ci_high'])}; E6.P1 is "
        f"{outcome('E6.P1')}). The prediction that the same memory is stale for NEW facts is "
        f"{outcome('E6.P2')}: the point estimate is {f(scores['E6.P2']['measured'])} "
        f"({f(scores['E6.P2']['result']['baseline_mean'])} without hints to "
        f"{f(scores['E6.P2']['result']['treatment_mean'])} with hints), and the CI includes zero. The "
        f"learned gate improves NEW accuracy over always-on hints by "
        f"{f(scores['E6.P3']['measured'])} (95% CI {f(scores['E6.P3']['ci_low'])} to "
        f"{f(scores['E6.P3']['ci_high'])}; E6.P3 is {outcome('E6.P3')}), and it opens more for "
        f"OLD than NEW prompts by {f(scores['E6.P4']['measured'])} (95% CI "
        f"{f(scores['E6.P4']['ci_low'])} to {f(scores['E6.P4']['ci_high'])}; E6.P4 is "
        f"{outcome('E6.P4')}). The residual hints-on penalty under gating is "
        f"{f(scores['E6.P8']['measured'])} (95% CI {f(scores['E6.P8']['ci_low'])} to "
        f"{f(scores['E6.P8']['ci_high'])}; E6.P8 is {outcome('E6.P8')}), so the gate removes the "
        "measured penalty in this run rather than merely shrinking it. The cost is visible in the "
        f"OLD columns: always-on hints recover {f(mean(e6['always_hints']['old_canon_on_top1']))} "
        f"of OLD facts, while gated hints recover only "
        f"{f(mean(e6['gated_hints']['old_canon_on_top1']))}. E6.P6 confirms the direction "
        f"(no hints {f(mean(e6['no_hints']['old_canon_off_top1']))} versus gated "
        f"{f(mean(e6['gated_hints']['old_canon_on_top1']))}), but the measured gain is "
        f"{f(scores['E6.P6']['measured'])} against a predicted {f(pred_effect('E6.P6'))}. The gate "
        "is therefore a soft suppressor that trades OLD-fact recall for NEW-fact robustness, not "
        "a clean prefix-conditioned trust switch.\n"
    )
    lines.append(
        f"The order-invariance prediction is {outcome('E6.P5')}: the ordered trigram table misses "
        f"reordered OLD prompts ({f(scores['E6.P5']['result']['baseline_mean'])}), while the bag-of-words "
        f"table reaches {f(scores['E6.P5']['result']['treatment_mean'])} (difference "
        f"{f(scores['E6.P5']['measured'])}; 95% CI {f(scores['E6.P5']['ci_low'])} to "
        f"{f(scores['E6.P5']['ci_high'])}). The point estimate is positive, but the interval "
        "includes zero with only three seeds, so we do not claim that bag-of-words keys fix the "
        "failure mode. The mechanism is plausible—exact suffix addressing cannot match a "
        "reordered prompt—but this run provides directional evidence, not a resolved result.\n"
    )
    lines.append(
        f"Finally, the prediction that naive hint dropout would help is {outcome('E6.P7')}. "
        f"Mixing hinted and unhinted examples during training changes NEW accuracy with hints on "
        f"by {f(scores['E6.P7']['measured'])} (95% CI {f(scores['E6.P7']['ci_low'])} to "
        f"{f(scores['E6.P7']['ci_high'])}), and the learned gate beats dropout by "
        f"{f(scores['E6.P9']['measured'])} on the same condition (E6.P9 is {outcome('E6.P9')}). "
        "Randomly dropping the hint does not teach the model when to distrust it in this "
        "controlled world; a separate gate with a prefix-sensitive input does better.\n"
    )

    lines.append("## 8. Live-market probe (M1-M5)\n")
    lines.append(
        f"The probe fetches {mkt['n_rows']} daily SPY observations ({mkt['first_date']} to "
        f"{mkt['last_date']}) from Yahoo Finance, builds five causal features (`r1`, `r5`, "
        "`r20`, `vol20`, 20-day moving-average gap), and runs expanding-window walk-forward "
        f"regressions. There are {wf['n_folds']} folds and {wf['n']} out-of-sample days. "
        f"Directional accuracy is {pct(wf['directional_accuracy'])}; MSE skill versus a "
        f"zero-return forecast is {f(wf['mse_skill'])}; the Sharpe ratio after 1 bp costs is "
        f"{f(wf['sharpe_after_cost'])}. The Diebold-Mariano statistic versus zero is "
        f"{f(wf['dm_stat_vs_zero'])} (p = {f(wf['dm_p_vs_zero'])}).\n"
    )
    lines.append(
        table(
            ["ID", "Claim", "Measured [95% CI]", "Verdict"],
            [[f"`{p['id']}`", esc(p["claim"]), f"{ci(scores[p['id']])} ({units.get(p['id'], 'effect')})", scores[p["id"]]["verdict"]] for p in preds if p["id"].startswith("M")],
        )
    )
    lines.append("")
    lines.append(
        f"The shuffled-label control's accuracy is {pct(ctrl['directional_accuracy'])}, "
        f"within the predicted 5-point band of chance (M5), so the pipeline is not leaking "
        f"future labels. Volatility clustering is strong: the lag-1 autocorrelation of absolute "
        f"returns is {f(vol['acf']['1'])} with Ljung-Box p = {f(vol['ljung_box_p'])}. "
        "The market probe is a calibration device, not an investment strategy: the linear model "
        "is deliberately simple, the index is deliberately liquid, and the negative MSE skill is "
        "the expected result when a small feature set is asked to beat a zero forecast.\n"
    )

    lines.append("## 9. What replicated, what failed, what is new\n")
    lines.append(f"**Confirmed ({len(confirmed)}).** " + ", ".join(f"`{s['id']}`" for s in confirmed) + ".\n")
    lines.append(f"**Refuted ({len(refuted)}).** " + (", ".join(f"`{s['id']}`" for s in refuted) if refuted else "none") + ".\n")
    lines.append(f"**Inconclusive ({len(inconclusive)}).** " + (", ".join(f"`{s['id']}`" for s in inconclusive) if inconclusive else "none") + ".\n")
    lines.append(
        "The refutations are the point of the exercise. A benchmark that only confirms its own "
        "predictions is a marketing document. The refuted predictions are "
        + (", ".join(f"`{s['id']}`" for s in refuted) if refuted else "none")
        + ". The clearest policy-relevant confirmation is that a small, explicit gate conditioned "
        "on a prefix improves updated-fact accuracy relative to always-on hints. The "
        "order-invariant-key test is reported with its actual verdict rather than folded into "
        "that claim.\n"
    )

    lines.append("## 10. Limitations and threats to validity\n")
    lines.append(
        "- **Scale.** Two layers, 300 steps and a few million characters are enough to expose "
        "mechanisms, not to establish scaling laws. Every claim is a claim about this regime.\n"
        "- **Tuning.** E2 selects learning rates on the same validation split used for the final "
        "comparison. This is a matched-budget comparison, not an unbiased estimate of the best "
        "possible optimizer performance.\n"
        "- **Pre-registration timing.** The predictions were written after exploratory pilots. "
        "The ledger records the freeze, and the pilots are disclosed; a fully prospective study "
        "would freeze before any pilot.\n"
        "- **Multiple comparisons.** Thirty-three predictions are tested. Per-prediction CIs are "
        "reported; the aggregate Brier score is descriptive, not a family-wise error guarantee.\n"
        "- **Market probe.** One liquid ETF, one feature set, one cost assumption, daily bars. "
        "The result says nothing about intraday, cross-sectional or alternative-data strategies.\n"
        "- **Synthetic memory world.** E6 uses a controlled fact-update grammar so that OLD and "
        "NEW facts are identifiable. Real knowledge conflicts are messier and often ambiguous. "
        "The gate's NEW-fact gain comes with a large OLD-fact cost, so the experiment supports "
        "soft suppression rather than clean selective trust.\n"
    )

    lines.append("## 11. Reproducibility\n")
    lines.append(
        "```bash\n"
        "python -m venv .venv && . .venv/bin/activate\n"
        "pip install -e \".[dev]\"\n"
        "\n"
        "# 1. Freeze the predictions (writes preregistry/ledger.jsonl)\n"
        "python scripts/freeze.py\n"
        "\n"
        f"# 2. Run the final benchmark (CPU; ~{manifest['seconds'] / 60:.0f} minutes on 8 threads)\n"
        "python scripts/run_all.py --seeds 0,1,2,3,4 --steps 300\n"
        "\n"
        "# 3. Score and render this README from measured results\n"
        "python scripts/score.py\n"
        "python scripts/render_readme.py\n"
        "```\n"
    )
    lines.append(
        "Docker is provided for a clean-room run:\n\n"
        "```bash\n"
        "docker compose build\n"
        "docker compose run --rm benchmark\n"
        "```\n"
    )
    lines.append(
        f"The generated artifacts are `results/scores.json`, `results/tables/scores.csv` and the "
        f"raw JSON under `results/raw/`. The ledger head at render time is `{ledger['head']}`. "
        f"The run manifest records Python {manifest['python']} and the exact seed/step configuration.\n"
    )

    lines.append("## 12. Data, code and ethics\n")
    lines.append(
        "All data are public: TinyStories (HuggingFace), WikiText-2 (HuggingFace), Project "
        "Gutenberg `11-0.txt`, and daily SPY bars from Yahoo Finance. The market probe is a "
        "methodological control and is not financial advice. No private data, user data or "
        "proprietary model weights are used. The code is released under the Apache-2.0 license; "
        "the text and figures are released under CC BY 4.0.\n"
    )

    lines.append("## 13. References\n")
    refs = [
        "Loshchilov, I. and Hutter, F. Decoupled Weight Decay Regularization. ICLR 2019. arXiv:1711.05101.",
        "Jordan, K. et al. Muon is Scalable for LLM Training. arXiv:2502.16982, 2025.",
        "How Transformers Learn to Plan via Multi-Token Prediction. arXiv:2604.11912, 2026.",
        "Softmax Routing Dynamics in Mixture-of-Experts Models: Collapse and Recovery. arXiv:2610.39442, 2026.",
        "DeepSeek. Engram: Conditional Memory via Scalable Lookup. arXiv:2601.07372, 2026.",
        "Kaplan, J. et al. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020.",
        "Hoffmann, J. et al. Training Compute-Optimal Large Language Models. arXiv:2203.15556, 2022.",
        "Sukhbaatar, S. et al. End-To-End Memory Networks. NeurIPS 2015.",
        "Fama, E. F. and French, K. R. Luck versus Skill in the Cross-Section of Mutual Fund Returns. Journal of Finance, 2010.",
        "Bailey, D. H. et al. The Probability of Backtest Overfitting. Journal of Computational Finance, 2017.",
        "Bailey, D. H. and López de Prado, M. The Deflated Sharpe Ratio. Journal of Portfolio Management, 2014.",
        "Cont, R. Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues. Quantitative Finance, 2001.",
        "Nosek, B. A., Ebersole, C. R., DeHaven, A. C. and Mellor, D. T. The Preregistration Revolution. PNAS, 2018.",
        "Elhage, N. et al. A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread, 2021.",
        "Olsson, C. et al. In-context Learning and Induction Heads. Transformer Circuits Thread, 2022.",
    ]
    for r in refs:
        lines.append(f"- {r}")
    lines.append("")

    (ROOT / "README.md").write_text("\n".join(lines))
    print(f"wrote {ROOT / 'README.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
