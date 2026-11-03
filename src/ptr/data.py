from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import requests
import torch

from .tokenizer import CharTokenizer, WordTokenizer

DEFAULT_URLS = {
    "tiny_stories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
    "wikitext2": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "gutenberg": "https://www.gutenberg.org/files/11/11-0.txt",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str | Path, retries: int = 3, timeout: int = 120) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
                return dest
        except Exception as exc:  # pragma: no cover - network
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def _read_parquet_text(path: Path) -> str:
    table = pq.read_table(path)
    column = "text" if "text" in table.column_names else table.column_names[0]
    return "\n".join(str(x) for x in table.column(column).to_pylist())


def load_public_text(
    name: str,
    cache_dir: str | Path = "data/cache",
    max_chars: int = 1_200_000,
    url: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Download and cache a public corpus; return text + provenance manifest."""
    if name not in DEFAULT_URLS and url is None:
        raise ValueError(f"unknown corpus {name}")
    url = url or DEFAULT_URLS[name]
    suffix = ".parquet" if url.endswith(".parquet") else ".txt"
    dest = Path(cache_dir) / f"{name}{suffix}"
    download(url, dest)
    if suffix == ".parquet":
        text = _read_parquet_text(dest)
    else:
        text = dest.read_text(encoding="utf-8", errors="replace")
    text = text[:max_chars]
    manifest = {
        "name": name,
        "url": url,
        "sha256": sha256_file(dest),
        "bytes": str(dest.stat().st_size),
        "chars_used": str(len(text)),
    }
    return text, manifest


def split_text(text: str, train_frac: float = 0.9) -> tuple[str, str]:
    cut = int(len(text) * train_frac)
    return text[:cut], text[cut:]


def char_dataset(
    text: str,
    ctx: int,
    train_frac: float = 0.9,
    tokenizer: CharTokenizer | None = None,
) -> dict:
    train_text, val_text = split_text(text, train_frac)
    tok = tokenizer or CharTokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return {"train": train_ids, "val": val_ids, "tokenizer": tok, "ctx": ctx}


def word_dataset(
    text: str,
    ctx: int,
    train_frac: float = 0.9,
    tokenizer: WordTokenizer | None = None,
) -> dict:
    train_text, val_text = split_text(text, train_frac)
    tok = tokenizer or WordTokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return {"train": train_ids, "val": val_ids, "tokenizer": tok, "ctx": ctx}


def sample_batch(ids: torch.Tensor, batch_size: int, ctx: int, generator: torch.Generator):
    if ids.numel() <= ctx + 2:
        raise ValueError("corpus too small for context length")
    ix = torch.randint(0, ids.numel() - ctx - 1, (batch_size,), generator=generator)
    x = torch.stack([ids[i : i + ctx] for i in ix])
    y = torch.stack([ids[i + 1 : i + ctx + 1] for i in ix])
    return x, y


def fixed_batches(ids: torch.Tensor, batch_size: int, ctx: int, n_batches: int, seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    return [sample_batch(ids, batch_size, ctx, g) for _ in range(n_batches)]


def manifest_rows(manifests: Iterable[dict]) -> list[dict]:
    return [dict(m) for m in manifests]


def save_manifest(manifests: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifests, indent=2, sort_keys=True))
