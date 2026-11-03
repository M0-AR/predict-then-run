from __future__ import annotations

import re
from collections import Counter


class CharTokenizer:
    """Byte-ish character tokenizer with a stable vocabulary."""

    def __init__(self, text: str, min_freq: int = 1):
        counts = Counter(text)
        chars = sorted([c for c, n in counts.items() if n >= min_freq])
        self.itos = ["<pad>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = 0
        self.vocab_size = len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "" for i in ids)


class WordTokenizer:
    """Simple whitespace/punctuation word tokenizer for the hint experiment."""

    _split = re.compile(r"\w+|[^\w\s]")

    def __init__(self, text: str, min_freq: int = 2, max_vocab: int = 6000):
        counts = Counter(self.tokenize(text))
        words = [w for w, n in counts.most_common(max_vocab) if n >= min_freq]
        self.itos = ["<pad>", "<unk>"] + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = 0
        self.unk_id = 1
        self.vocab_size = len(self.itos)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return cls._split.findall(text.lower())

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(w, self.unk_id) for w in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids)
