#!/usr/bin/env python3
"""Record the interactive preview and encode the README demo video.

Requires Playwright and ffmpeg:

    python -m pip install playwright
    python -m playwright install chromium
    python scripts/record_demo.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "preview.html"
ASSETS = ROOT / "assets"
VIDEO_SIZE = {"width": 1280, "height": 720}


def pause(page, milliseconds: int) -> None:
    page.wait_for_timeout(milliseconds)


def scroll_to(page, selector: str, milliseconds: int = 1400) -> None:
    page.evaluate(
        "(selector) => document.querySelector(selector).scrollIntoView({behavior: 'smooth', block: 'start'})",
        selector,
    )
    pause(page, milliseconds)


def record(raw_dir: Path) -> Path:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport=VIDEO_SIZE,
            record_video_dir=str(raw_dir),
            record_video_size=VIDEO_SIZE,
        )
        page = context.new_page()
        page.goto(PREVIEW.as_uri(), wait_until="load")
        pause(page, 1800)

        scroll_to(page, "#evidence", 1600)
        page.get_by_role("button", name="Refuted", exact=True).click()
        pause(page, 1200)
        page.locator("#claims-body details summary").first.click()
        pause(page, 1600)
        page.get_by_role("button", name="All", exact=True).click()
        pause(page, 500)

        scroll_to(page, "#findings", 1300)
        scroll_to(page, "#learn", 1300)
        page.locator('.quiz-option[data-option="1"]').click()
        pause(page, 1700)

        scroll_to(page, "#business .biz-grid", 1300)
        page.locator("#calc-studies").evaluate(
            "(element) => { element.value = '60'; element.dispatchEvent(new Event('input', {bubbles: true})); }"
        )
        pause(page, 1700)

        scroll_to(page, "#repo .repo-grid", 1500)
        pause(page, 1800)

        context.close()
        browser.close()

    videos = sorted(raw_dir.glob("*.webm"))
    if not videos:
        raise RuntimeError("Playwright did not produce a video")
    return videos[-1]


def encode(source: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the demo")

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(ASSETS / "demo.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-vf",
            "fps=8,scale=800:-1:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5",
            str(ASSETS / "demo.gif"),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    if not PREVIEW.exists():
        raise SystemExit("preview.html is missing; run python scripts/build_site.py first")
    ASSETS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ptr-demo-") as raw:
        source = record(Path(raw))
        encode(source)
    for name in ("demo.mp4", "demo.gif"):
        path = ASSETS / name
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
