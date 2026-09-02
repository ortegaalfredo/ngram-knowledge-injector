"""Tokenization for the PLE injector.

Two backends, in preference order:
  1. HF `tokenizers` (needs tokenizer.json) -- fast, exact.
  2. A GPT-2 BPE rebuilt from the GGUF's own tokenizer.ggml.tokens + merges --
     fully offline, and it is exactly the vocabulary llama.cpp uses at inference,
     which is what matters for reproducing the runtime's n-gram rows.

Both are validated to agree on ordinary text (they can only differ on unused
control slots, which never appear in injected knowledge).
"""
from __future__ import annotations

import functools
import os
import re
from typing import Sequence

import numpy as np

try:
    import regex as _re
except Exception:  # pragma: no cover
    import re as _re  # type: ignore

# GPT-2 pre-tokenizer pattern (the qwen35 pre-tokenizer uses the same scheme).
GPT2PAT = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Qwen (qwen2/qwen35) pre-tokenizer regex, taken verbatim from tokenizer.json.
QWEN35PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|"
    r"\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class GGUFTokenizer:
    """GPT-2 style BPE reconstructed from GGUF metadata (offline)."""

    def __init__(self, tokens: Sequence[str], merges: Sequence[str],
                 byte_fallback: bool = False, add_prefix_space: bool = False,
                 pattern: str = QWEN35PAT):
        self.byte_fallback = byte_fallback
        self.decoder = bytes_to_unicode()
        self.encoder = {v: k for k, v in self.decoder.items()}
        # GGUF stores tokens as raw strings where a leading space is 'Ġ' (U+0120)
        # already in the GPT-2 unicode-mapped space.
        self.tok_to_id = {t: i for i, t in enumerate(tokens)}
        self.tokens = list(tokens)
        self.pattern = pattern
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for rank, pair in enumerate(merges):
            a, _, b = pair.partition(" ")
            self.bpe_ranks[(a, b)] = rank
        self.cache: dict[str, list[str]] = {}

    @classmethod
    def from_gguf(cls, path: str) -> "GGUFTokenizer":
        # ple_core.reader() caches the parsed file (seconds per parse on a real
        # vocab); share it instead of building a second GGUFReader.
        from ple_core import reader
        r = reader(path)

        def cont(f):
            c = f.contents
            return c() if callable(c) else c

        def get(k, default=None):
            return cont(r.fields[k]) if k in r.fields else default

        tokens = list(get("tokenizer.ggml.tokens", []))
        if not tokens:
            raise ValueError("tokenizer.ggml.tokens missing from GGUF")
        merges = [m.decode() if isinstance(m, (bytes, bytearray)) else m
                  for m in get("tokenizer.ggml.merges", [])]
        bf = bool(get("tokenizer.ggml.byte_fallback", False))
        aps = bool(get("tokenizer.ggml.add_prefix_space", False))
        pre = get("tokenizer.ggml.pre", "qwen2")
        pattern = GPT2PAT if pre in ("gpt2", "bpe") else QWEN35PAT
        return cls(tokens, merges, byte_fallback=bf, add_prefix_space=aps, pattern=pattern)

    def _bpe(self, text: str) -> list[str]:
        if text in self.cache:
            return self.cache[text]
        # GPT-2 maps each utf-8 byte through the byte->unicode table, then BPEs.
        word = "".join(self.decoder[b] for b in text.encode("utf-8"))
        symbols = list(word)
        if len(symbols) == 1:
            self.cache[text] = symbols
            return symbols
        while True:
            pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
            best = None
            bestrank = None
            for p in pairs:
                r = self.bpe_ranks.get(p)
                if r is not None and (bestrank is None or r < bestrank):
                    best, bestrank = p, r
            if best is None:
                break
            first, second = best
            i = 0
            new = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(symbols[i])
                    i += 1
            symbols = new
            if len(symbols) == 1:
                break
        self.cache[text] = symbols
        return symbols

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        for piece in _re.findall(self.pattern, text):
            for sym in self._bpe(piece):
                tid = self.tok_to_id.get(sym)
                if tid is None and self.byte_fallback:
                    for b in sym.encode("utf-8"):
                        tid2 = self.tok_to_id.get(self.decoder[b])
                        if tid2 is not None:
                            ids.append(tid2)
                    continue
                if tid is not None:
                    ids.append(tid)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            if 0 <= i < len(self.tokens):
                t = self.tokens[i]
                out.append("".join(chr(self.encoder.get(c, ord(c))) for c in t))
        return "".join(out).encode("utf-8", "ignore").decode("utf-8", "replace")


class HFTokenizer:
    """HF `tokenizers` backend (needs tokenizer.json)."""

    def __init__(self, tokenizer_json_path: str):
        from tokenizers import Tokenizer  # lazy import
        self.tk = Tokenizer.from_file(tokenizer_json_path)

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return self.tk.encode(text, add_special_tokens=add_special).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=False)


@functools.lru_cache(maxsize=4)
def _gguf_tok_cached(gguf_path: str) -> "GGUFTokenizer":
    return GGUFTokenizer.from_gguf(gguf_path)


def make_tokenizer(gguf_path: str, tokenizer_json: str | None = None,
                   prefer: str = "auto"):
    """Return a tokenizer. prefer in {auto, hf, gguf}."""
    if tokenizer_json is None:
        env = os.environ.get("PLE_TOKENIZER_JSON")
        tokenizer_json = env if env and os.path.exists(env) else None
    if prefer in ("auto", "hf") and tokenizer_json:
        try:
            return HFTokenizer(tokenizer_json), "hf"
        except Exception:
            if prefer == "hf":
                raise
    return _gguf_tok_cached(gguf_path), "gguf"


def validate_tokenizers(gguf_path: str, tokenizer_json: str,
                        samples: Sequence[str]) -> tuple[bool, list[tuple[str, list[int], list[int]]]]:
    """Compare HF vs GGUF BPE on sample strings; return (all_agree, diffs)."""
    g = GGUFTokenizer.from_gguf(gguf_path)
    h = HFTokenizer(tokenizer_json)
    diffs = []
    for s in samples:
        gi, hi = g.encode(s), h.encode(s)
        if gi != hi:
            diffs.append((s, gi, hi))
    return (len(diffs) == 0), diffs
