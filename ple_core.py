"""Core primitives for injecting knowledge into a Qwen3.8-Flash-Next (qwen4exp)
PLE n-gram embedding table inside a GGUF.

Everything here is derived from the reference implementation
(unslothai/llama.cpp, branch qwen4exp/qwen3.8-flash-next, src/models/qwen4exp.cpp,
llm_graph_input_ple::set_input) and verified against it.

    mixed_n = (ctx[0]*m[0]) ^ (ctx[1]*m[1]) ^ ... ^ (ctx[n-1]*m[n-1])   # uint64
    row     = mixed_n % head_vocab_sizes[h] + head_offsets[h]

Table tensor: per_layer_token_embd.weight, shape [row_dim, n_rows] (row-major,
so table[r] is the contiguous slice [r*row_dim : (r+1)*row_dim]).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as Q
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize, quantize

M64 = (1 << 64) - 1

# KV keys, matching llama-arch.cpp on the qwen4exp branch.
KV_PLE_LAYERS = "{arch}.ple.layers"
KV_PLE_NGRAM_SIZE = "{arch}.ple.ngram_size"
KV_PLE_HEADS_PER_NGRAM = "{arch}.ple.heads_per_ngram"
KV_PLE_CONV_KERNEL = "{arch}.ple.conv_kernel"
KV_PLE_LAYER_MULTIPLIERS = "{arch}.ple.layer_multipliers"
KV_PLE_HEAD_OFFSETS = "{arch}.ple.head_offsets"
KV_PLE_HEAD_VOCAB_SIZES = "{arch}.ple.head_vocab_sizes"
KV_PLE_EOS_TOKEN_ID = "{arch}.ple.eos_token_id"
KV_PLE_IMAGE_TOKEN_ID = "{arch}.ple.image_token_id"
KV_EMBD_PER_LAYER = "{arch}.embedding_length_per_layer_input"

TABLE_TENSOR = "per_layer_token_embd.weight"


def _contents(field_obj):
    c = field_obj.contents
    return c() if callable(c) else c


# --------------------------------------------------------------------------- #
# hash constants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PleConstants:
    ngram_size: int
    heads_per_ngram: int
    multipliers: tuple[int, ...]
    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]
    eos_token_id: int
    image_token_id: int | None
    row_dim: int
    ple_layers: tuple[int, ...]
    conv_kernel: int

    @property
    def n_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def n_rows(self) -> int:
        return self.head_offsets[-1] + self.head_vocab_sizes[-1]

    def head_of(self, h: int) -> tuple[int, int]:
        return self.head_offsets[h], self.head_vocab_sizes[h]

    def head_ranges(self) -> list[tuple[int, int, int]]:
        """[(order, first_head, head_count)] order 2..ngram_size."""
        out = []
        for n in range(2, self.ngram_size + 1):
            out.append((n, (n - 2) * self.heads_per_ngram, self.heads_per_ngram))
        return out


def load_ple_constants(metadata: dict[str, object]) -> PleConstants:
    """Build PleConstants from a {key: value} metadata dict of one GGUF shard."""
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str):
        raise ValueError("general.architecture missing; is this a GGUF model file?")

    def get(key: str, required: bool = True, default=None):
        full = arch + key if key.startswith(".") else key
        v = metadata.get(full)
        if v is None and required:
            raise ValueError(f"required metadata key missing: {full}")
        return default if v is None else v

    ngram_size = int(get(".ple.ngram_size"))
    heads = int(get(".ple.heads_per_ngram"))
    mult = tuple(int(x) for x in get(".ple.layer_multipliers"))
    offs = tuple(int(x) for x in get(".ple.head_offsets"))
    vsz = tuple(int(x) for x in get(".ple.head_vocab_sizes"))
    layers = tuple(int(x) for x in get(".ple.layers"))
    row_dim = int(get(".embedding_length_per_layer_input"))
    eos = int(get(".ple.eos_token_id"))
    img = get(".ple.image_token_id", required=False)

    if len(mult) < ngram_size:
        raise ValueError(f"layer_multipliers has {len(mult)} entries, need ngram_size={ngram_size}")
    if len(offs) != (ngram_size - 1) * heads or len(vsz) != (ngram_size - 1) * heads:
        raise ValueError("head_offsets/head_vocab_sizes length != (ngram_size-1)*heads_per_ngram")
    # offsets must be a running prefix-sum of vocab sizes: the runtime indexes one
    # flat table with (mixed % vsz[h]) + offs[h], so a gap would silently alias.
    expect = 0
    for h, (o, v) in enumerate(zip(offs, vsz)):
        if o != expect:
            raise ValueError(f"head_offsets[{h}]={o} is not the prefix sum {expect}; "
                             "heads must tile the table contiguously")
        expect = o + v
    return PleConstants(
        ngram_size=ngram_size, heads_per_ngram=heads, multipliers=mult,
        head_offsets=offs, head_vocab_sizes=vsz, eos_token_id=eos,
        image_token_id=int(img) if img is not None else None,
        row_dim=row_dim, ple_layers=layers,
        conv_kernel=int(get(".ple.conv_kernel", required=False, default=4)),
    )


# --------------------------------------------------------------------------- #
# the hash (must match the C++ reference exactly)
# --------------------------------------------------------------------------- #
def mixed_value(ctx: Sequence[int], n: int, multipliers: Sequence[int]) -> int:
    """mixed_n for the n-gram ctx[0..n-1] (ctx[0] is the newest token)."""
    mixed = (int(ctx[0]) * multipliers[0]) & M64
    for j in range(1, n):
        mixed ^= (int(ctx[j]) * multipliers[j]) & M64
    return mixed


def rows_for_token(tok: int, prev: Sequence[int], c: PleConstants) -> list[int]:
    """The n_heads table rows for one token given up to (ngram_size-1) predecessors.

    prev is oldest-first like the runtime's prev[] buffer, i.e. for ngram_size=3
    prev == [t-2, t-1]. Missing predecessors must be passed as the eos token id
    (the runtime substitutes eos when a predecessor is absent or cut by an eos).
    """
    n_gram = c.ngram_size
    ctx = [int(tok)]
    cut = False
    for s in range(1, n_gram):
        # once an eos (or a missing predecessor) is seen, every OLDER slot is eos
        t = c.eos_token_id if cut or not prev or len(prev) < s \
            else int(prev[len(prev) - s])
        cut = cut or t < 0 or t == c.eos_token_id
        ctx.append(c.eos_token_id if cut else t)
    rows: list[int] = []
    for n in range(2, n_gram + 1):
        mixed = mixed_value(ctx, n, c.multipliers)
        base = (n - 2) * c.heads_per_ngram
        for g in range(c.heads_per_ngram):
            h = base + g
            off, vsz = c.head_offsets[h], c.head_vocab_sizes[h]
            rows.append(int(mixed % vsz) + off)
    return rows


def context_windows(tokens: Sequence[int], c: PleConstants) -> list[list[tuple[int, tuple[int, ...]]]]:
    """Per position: list of (order, ctx) n-grams the runtime will hash.

    Mirrors the runtime's eos reset: everything at or before an eos is cut, and the
    eos of the token itself does not cut its own context.
    """
    out = []
    for i, tok in enumerate(tokens):
        per_pos = []
        for n in range(2, c.ngram_size + 1):
            ctx = [int(tok)]
            cut = False
            for s in range(1, n):
                j = i - s
                t = int(tokens[j]) if j >= 0 else c.eos_token_id
                if cut or t < 0 or t == c.eos_token_id:
                    cut = True
                    ctx.append(c.eos_token_id)
                else:
                    ctx.append(t)
            per_pos.append((n, tuple(ctx)))
        out.append(per_pos)
    return out


def rows_for_sequence(tokens: Sequence[int], c: PleConstants) -> list[list[int]]:
    """rows[i] = the n_heads rows the runtime hashes at position i."""
    res = []
    n_prev = c.ngram_size - 1
    for i, tok in enumerate(tokens):
        prev = [int(tokens[j]) for j in range(max(0, i - n_prev), i)]
        res.append(rows_for_token(int(tok), prev, c))
    return res


# --------------------------------------------------------------------------- #
# quantized row addressing
# --------------------------------------------------------------------------- #
def bytes_per_row(qtype: Q, row_dim: int) -> int:
    blck, bps = GGML_QUANT_SIZES[qtype]
    if row_dim % blck:
        raise ValueError(f"{qtype.name} needs row_dim divisible by {blck}, got {row_dim}")
    return row_dim // blck * bps


def is_fully_quantized(qtype: Q, row_dim: int) -> bool:
    blck, _ = GGML_QUANT_SIZES[qtype]
    return row_dim % blck == 0


class RowCodec:
    """Encode/decode single rows of a quantized table, block-aligned."""

    def __init__(self, qtype: Q, row_dim: int):
        self.qtype = qtype
        self.row_dim = row_dim
        self.bpr = bytes_per_row(qtype, row_dim)
        self.blck, _ = GGML_QUANT_SIZES[qtype]
        self.n_blocks = row_dim // self.blck
        if not is_fully_quantized(qtype, row_dim):
            raise ValueError(f"{qtype.name} with row_dim={row_dim} is not block aligned")
        try:
            quantize(np.zeros(self.blck, dtype=np.float32), qtype)
        except Exception as exc:  # gguf-py raises NotImplementedError
            raise ValueError(
                f"gguf-py cannot quantize to {qtype.name}; pick a GGUF whose table type is "
                "supported by gguf.quants.quantize (e.g. Q8_0, F32, F16, Q4_0, IQ4_XS)."
            ) from exc

    def decode(self, raw: np.ndarray) -> np.ndarray:
        raw = np.ascontiguousarray(raw, dtype=np.uint8)
        if raw.size != self.bpr:
            raise ValueError(f"row slice is {raw.size} bytes, expected {self.bpr}")
        if self.qtype == Q.F32:
            return raw.view(np.float32).astype(np.float32)
        if self.qtype == Q.F16:
            return raw.view(np.float16).astype(np.float32)
        return dequantize(raw.reshape(self.n_blocks, -1), self.qtype).reshape(self.row_dim)

    def encode(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(self.row_dim)
        if self.qtype == Q.F32:
            return vec.astype(np.float32).view(np.uint8)
        if self.qtype == Q.F16:
            return vec.astype(np.float16).view(np.uint8)
        enc = quantize(vec.reshape(1, self.row_dim), self.qtype)
        return np.ascontiguousarray(enc, dtype=np.uint8).reshape(-1)


# --------------------------------------------------------------------------- #
# GGUF reading
# --------------------------------------------------------------------------- #
@dataclass
class TableLocation:
    path: str
    name: str
    shard_no: int
    split_count: int
    data_offset: int
    n_rows: int
    row_dim: int
    qtype: Q
    n_bytes: int


@dataclass
class ShardInfo:
    path: str
    shard_no: int
    n_tensors: int
    size: int


# Parsing a GGUF header costs seconds per file: tokenizer.ggml.tokens/.merges
# are ~300k array elements and GGUFReader walks every one of them. Every
# consumer (metadata, table lookup, tokenizer) shares ONE parse per file via
# reader(); mtime+size key the cache so a modified model is re-parsed.
@functools.lru_cache(maxsize=8)
def _reader_cached(path: str, mtime: float, size: int) -> GGUFReader:
    return GGUFReader(path)


def reader(path: str) -> GGUFReader:
    st = os.stat(path)
    return _reader_cached(path, st.st_mtime, st.st_size)


def read_metadata(path: str) -> dict[str, object]:
    r = reader(path)
    return {k: _contents(v) for k, v in r.fields.items()}


@functools.lru_cache(maxsize=8)
def _discover_shards_cached(first_path: str) -> tuple:
    return tuple(_discover_shards_uncached(first_path))


def discover_shards(first_path: str) -> list:
    return list(_discover_shards_cached(first_path))


def _discover_shards_uncached(first_path: str) -> list:
    """List every shard of a (possibly split) GGUF, ordered by split.no."""
    meta = read_metadata(first_path)
    split_count = int(meta.get("split.count", 1))
    directory = os.path.dirname(os.path.abspath(first_path))
    base = os.path.basename(first_path)
    if split_count <= 1:
        return [ShardInfo(first_path, 0, _count_tensors(first_path), os.path.getsize(first_path))]
    stem = _strip_shard_suffix(base)
    out: list[ShardInfo] = []
    for i in range(split_count):
        cand = _find_shard(directory, stem, i, split_count)
        if cand is None:
            raise FileNotFoundError(
                f"shard {i + 1}/{split_count} of {base} not found in {directory}")
        # do NOT open every shard here: a still-downloading shard can fail to parse.
        out.append(ShardInfo(cand, i, -1, os.path.getsize(cand)))
    return out


def _count_tensors(path: str) -> int:
    return len(reader(path).tensors)


def _strip_shard_suffix(name: str) -> str:
    import re
    # tolerate trailing junk after .gguf (e.g. "?download=true")
    m = re.search(r"[-.](\d{5})-of-(\d{5})\.gguf", name)
    if m:
        return name[: m.start()]
    m = re.search(r"\.(\d{5})\.gguf", name)
    if m:
        return name[: m.start()]
    m = re.search(r"\.gguf", name)
    return name[: m.start()] if m else name


def _find_shard(directory: str, stem: str, idx: int, total: int) -> str | None:
    import glob
    # Match on the shard index pattern so stray download suffixes
    # (e.g. wget leaving "?download=true") are tolerated.
    pats = [
        f"{stem}-{idx + 1:05d}-of-{total:05d}*",
        f"{stem}.{idx:05d}*",
    ]
    for pat in pats:
        hit = sorted(glob.glob(os.path.join(glob.escape(directory), pat)))
        if hit:
            return hit[0]
    return None


@functools.lru_cache(maxsize=8)
def _locate_table_cached(shard_paths: tuple) -> TableLocation:
    return _locate_table_uncached([ShardInfo(p, i, -1, 0) for i, p in enumerate(shard_paths)])


def locate_table(shards: Sequence[ShardInfo]) -> TableLocation:
    return _locate_table_cached(tuple(sh.path for sh in shards))


def _locate_table_uncached(shards: Sequence[ShardInfo]) -> TableLocation:
    for sh in shards:
        try:
            r = reader(sh.path)
        except Exception:
            # shard unreadable (e.g. still downloading); skip, the table may be elsewhere
            continue
        for t in r.tensors:
            if t.name == TABLE_TENSOR:
                dims = [int(x) for x in t.shape]
                row_dim, n_rows = dims[0], dims[1]
                return TableLocation(
                    path=sh.path, name=t.name, shard_no=sh.shard_no,
                    split_count=len(shards), data_offset=int(t.data_offset),
                    n_rows=n_rows, row_dim=row_dim, qtype=Q(int(t.tensor_type)),
                    n_bytes=int(t.n_bytes),
                )
    raise ValueError(f"tensor {TABLE_TENSOR} not found in any shard "
                     "(this GGUF has no n-gram/PLE table)")


def table_sha1_rows(loc: TableLocation, rows: Iterable[int]) -> str:
    """Checksum over specific rows, for overlay provenance."""
    h = hashlib.sha1()
    with open(loc.path, "rb") as f:
        for r in sorted(set(int(x) for x in rows)):
            f.seek(loc.data_offset + r * bytes_per_row(loc.qtype, loc.row_dim))
            h.update(f.read(bytes_per_row(loc.qtype, loc.row_dim)))
    return h.hexdigest()


@functools.lru_cache(maxsize=8)
def get_codec(qtype: Q, row_dim: int) -> RowCodec:
    """RowCodec construction probes the quantizer; reuse one per (qtype, dim)."""
    return RowCodec(qtype, row_dim)


def read_rows(loc: TableLocation, rows: Sequence[int]) -> np.ndarray:
    codec = get_codec(loc.qtype, loc.row_dim)
    out = np.zeros((len(rows), loc.row_dim), dtype=np.float32)
    with open(loc.path, "rb") as f:
        for i, r in enumerate(rows):
            f.seek(loc.data_offset + int(r) * codec.bpr)
            out[i] = codec.decode(np.frombuffer(f.read(codec.bpr), dtype=np.uint8))
    return out
