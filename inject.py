#!/usr/bin/env python3
"""Inject knowledge into the PLE n-gram embedding table of a Qwen3.8-Flash-Next
(qwen4exp) GGUF.

The n-gram table is a hash-addressed memory: at inference the runtime hashes the
local context (current token + predecessors) into `n_heads` rows of one shared
table and adds the gathered vectors into the residual stream. This tool computes
exactly those rows for the contexts you give it and writes a chosen vector into
them, so that whenever the model sees that context the injected signal fires.

Because the table is hash-addressed, a row is shared by every n-gram that collides
on it. The tool therefore never blindly clobbers: it reports collisions and, by
default, blends into the existing row instead of overwriting.

Input JSON (see examples/knowledge.json):
{
  "defaults": { "at": "last", "heads": "all", "orders": "all",
                "op": "blend", "alpha": 0.5, "scale": 1.0 },
  "entries": [
    { "trigger": "The capital of France is",
      "vector": [ ...160 floats... ] | "path.vec.npy" | "random" | "zero",
      "op": "blend|set|add|scale|copy_from",
      "alpha": 0.5, "scale": 1.0, "seed": 1234,
      "at": "last|all|first|<int>", "heads": [0,1,...] | "all",
      "orders": [2,3] | "all" }
  ]
}

Modes:
  --mode materialize : write a full updated GGUF (copies the table shard, hardlinks
                       the rest). Portable, works with any loader.
  --mode overlay     : write a tiny .plepatch sidecar (row_id -> raw bytes) plus a
                       manifest; pair with the llama.cpp patch to apply at load.
  --mode in-place    : patch the GGUF directly, writing an undo file first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ple_core import (  # noqa: E402
    PleConstants, RowCodec, TableLocation, discover_shards, load_ple_constants,
    locate_table, read_metadata, read_rows, rows_for_sequence, bytes_per_row,
)
from ple_tok import make_tokenizer  # noqa: E402

VALID_OPS = {"set", "blend", "add", "scale", "copy_from", "zero"}


# --------------------------------------------------------------------------- #
# knowledge spec
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    trigger: str
    vector: Any = None
    op: str = "blend"
    alpha: float = 0.5
    scale: float = 1.0
    seed: int | None = None
    at: str | int = "last"
    heads: str | list[int] = "all"
    orders: str | list[int] = "all"
    prefix: str = ""          # context placed before the trigger (affects hashing)
    copy_from: str | None = None
    note: str = ""

    @staticmethod
    def from_obj(o: dict, defaults: dict) -> "Entry":
        merged = {**defaults, **o}
        trig = merged.get("trigger")
        if not isinstance(trig, str) or trig == "":
            raise ValueError(f"entry needs a non-empty 'trigger': {o!r}")
        op = merged.get("op", "blend")
        if op not in VALID_OPS:
            raise ValueError(f"unknown op {op!r}; expected one of {sorted(VALID_OPS)}")
        return Entry(
            trigger=trig, vector=merged.get("vector"), op=op,
            alpha=float(merged.get("alpha", 0.5)), scale=float(merged.get("scale", 1.0)),
            seed=merged.get("seed"), at=merged.get("at", "last"),
            heads=merged.get("heads", "all"), orders=merged.get("orders", "all"),
            prefix=merged.get("prefix", ""), copy_from=merged.get("copy_from"),
            note=merged.get("note", ""),
        )


def load_knowledge(path: str) -> tuple[list[Entry], dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        entries_raw, defaults = data, {}
    else:
        entries_raw = data.get("entries", [])
        defaults = data.get("defaults", {}) or {}
    return [Entry.from_obj(o, defaults) for o in entries_raw], defaults


# --------------------------------------------------------------------------- #
# row selection
# --------------------------------------------------------------------------- #
def head_indices(entry: Entry, c: PleConstants) -> list[int]:
    if entry.heads == "all":
        return list(range(c.n_heads))
    hs = [int(h) for h in entry.heads]
    for h in hs:
        if not 0 <= h < c.n_heads:
            raise ValueError(f"head {h} out of range 0..{c.n_heads - 1}")
    return hs


def order_heads(entry: Entry, c: PleConstants) -> list[int]:
    """Head indices restricted to selected n-gram orders (2..ngram_size)."""
    orders = range(2, c.ngram_size + 1) if entry.orders == "all" else [int(o) for o in entry.orders]
    allowed: set[int] = set()
    for n in orders:
        if not 2 <= n <= c.ngram_size:
            raise ValueError(f"order {n} out of range 2..{c.ngram_size}")
        base = (n - 2) * c.heads_per_ngram
        allowed.update(range(base, base + c.heads_per_ngram))
    return [h for h in head_indices(entry, c) if h in allowed]


def positions(entry: Entry, n_tokens: int) -> list[int]:
    if entry.at == "all":
        return list(range(n_tokens))
    if entry.at == "last":
        return [n_tokens - 1] if n_tokens else []
    if entry.at == "first":
        return [0]
    return [int(entry.at)]


# --------------------------------------------------------------------------- #
# vector resolution
# --------------------------------------------------------------------------- #
def resolve_vector(entry: Entry, row_dim: int, base_dir: str,
                   existing: np.ndarray, loc: TableLocation,
                   tok, c: PleConstants, rows_at: dict[int, list[int]]) -> np.ndarray:
    """Return the target vector (row_dim,) for this entry. `existing` is the current
    row content the op applies to (for blend/add/scale)."""
    if entry.op == "zero":
        return np.zeros(row_dim, dtype=np.float32)
    if entry.op == "scale":
        return existing * np.float32(entry.scale)
    if entry.op == "copy_from":
        if not entry.copy_from:
            raise ValueError("copy_from op needs a 'copy_from' trigger string")
        src = tok.encode(entry.copy_from)
        src_rows = rows_for_sequence(src, c)
        if not src_rows:
            raise ValueError("copy_from trigger produced no tokens")
        # gather the source rows for the same head set at the source's last position
        hidx = order_heads(entry, c)
        src_last = src_rows[-1]
        vals = read_rows(loc, [src_last[h] for h in hidx])
        # average the source heads into one vector (heads are parallel views)
        return vals.mean(axis=0).astype(np.float32)
    v = entry.vector
    if v is None:
        raise ValueError(f"entry {entry.trigger!r} needs a 'vector' for op={entry.op}")
    if isinstance(v, str):
        low = v.lower()
        if low == "random":
            rng = np.random.default_rng(entry.seed)
            return (rng.standard_normal(row_dim) * 0.02).astype(np.float32)
        if low == "zero":
            return np.zeros(row_dim, dtype=np.float32)
        p = v if os.path.isabs(v) else os.path.join(base_dir, v)
        arr = load_vector_file(p)
    else:
        arr = np.asarray(v, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size != row_dim:
        raise ValueError(f"vector length {arr.size} != row_dim {row_dim}")
    return arr


def load_vector_file(p: str) -> np.ndarray:
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.endswith(".npy"):
        return np.load(p)
    if p.endswith(".json"):
        with open(p) as f:
            return np.asarray(json.load(f), dtype=np.float32)
    return np.fromfile(p, dtype=np.float32)


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    row_ops: dict[int, np.ndarray] = field(default_factory=dict)  # row -> final float vec
    touched: dict[int, int] = field(default_factory=dict)         # row -> #entries touching
    report: list[dict] = field(default_factory=list)


def build_plan(entries: list[Entry], tok, c: PleConstants, loc: TableLocation,
               base_dir: str) -> Plan:
    plan = Plan()
    for e in entries:
        text = e.prefix + e.trigger
        toks = tok.encode(text)
        if not toks:
            plan.report.append({"trigger": e.trigger, "skipped": "empty tokenization"})
            continue
        all_rows = rows_for_sequence(toks, c)
        hidx = order_heads(e, c)
        pos = [p for p in positions(e, len(toks)) if 0 <= p < len(toks)]
        target_rows = sorted({all_rows[p][h] for p in pos for h in hidx})
        if not target_rows:
            plan.report.append({"trigger": e.trigger, "skipped": "no target rows"})
            continue
        cur = read_rows(loc, target_rows)  # (k, row_dim)
        # apply per existing row so blend/scale see current content
        for j, r in enumerate(target_rows):
            existing = plan.row_ops.get(r, cur[j])
            tgt = resolve_vector(e, loc.row_dim, base_dir, existing, loc, tok, c, {})
            if e.op in ("set", "zero", "scale", "copy_from"):
                newv = tgt
            elif e.op == "blend":
                a = np.float32(e.alpha)
                newv = existing * (1 - a) + tgt * a
            elif e.op == "add":
                newv = existing + tgt * np.float32(e.scale)
            else:  # pragma: no cover
                raise ValueError(e.op)
            plan.row_ops[r] = np.asarray(newv, dtype=np.float32)
            plan.touched[r] = plan.touched.get(r, 0) + 1
        plan.report.append({
            "trigger": e.trigger, "op": e.op, "n_tokens": len(toks),
            "positions": pos, "heads": len(hidx), "orders":
                ("all" if e.orders == "all" else e.orders),
            "target_rows": len(target_rows),
        })
    return plan


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def sha1_rows(loc: TableLocation, rows) -> str:
    h = hashlib.sha1()
    bpr = bytes_per_row(loc.qtype, loc.row_dim)
    with open(loc.path, "rb") as f:
        for r in sorted(set(int(x) for x in rows)):
            f.seek(loc.data_offset + r * bpr)
            h.update(f.read(bpr))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# overlay format (fixed 128-byte header so it is trivial to parse in C)
#   0:8 magic "PLEOVLY1" | 8:2 ver | 10:2 flags | 12:4 qtype | 16:8 row_dim
#   24:8 bytes_per_row | 32:8 n_rows | 40:8 manifest_len | 48:8 reserved
#   56:64 tensor name (NUL padded) | 120:8 json follows | 128: rows
#   row = uint64 row_id + bytes_per_row raw bytes
OVERLAY_MAGIC = b"PLEOVLY1"
OVERLAY_VER = 1
OVERLAY_HDR = 128


def write_overlay(out_path: str, plan: Plan, loc: TableLocation, c: PleConstants,
                  entries: list[Entry], model_tag: str) -> None:
    codec = RowCodec(loc.qtype, loc.row_dim)
    rows = sorted(plan.row_ops)
    manifest = {
        "format": "ple-overlay-v1",
        "model": model_tag,
        "table_tensor": loc.name,
        "table_sha1_before": sha1_rows(loc, rows),
        "row_dim": loc.row_dim,
        "qtype": loc.qtype.name,
        "bytes_per_row": codec.bpr,
        "n_rows": len(rows),
        "ple": {"ngram_size": c.ngram_size, "heads_per_ngram": c.heads_per_ngram,
                "eos_token_id": c.eos_token_id},
        "entries": [{"trigger": e.trigger, "op": e.op, "note": e.note} for e in entries],
    }
    mbytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    name = loc.name.encode("utf-8")[:63]
    hdr = bytearray(OVERLAY_HDR)
    hdr[0:8] = OVERLAY_MAGIC
    hdr[8:10] = int(OVERLAY_VER).to_bytes(2, "little")
    hdr[12:16] = int(loc.qtype).to_bytes(4, "little")
    hdr[16:24] = int(loc.row_dim).to_bytes(8, "little")
    hdr[24:32] = int(codec.bpr).to_bytes(8, "little")
    hdr[32:40] = int(len(rows)).to_bytes(8, "little")
    hdr[40:48] = int(len(mbytes)).to_bytes(8, "little")
    hdr[56:56 + len(name)] = name
    with open(out_path, "wb") as f:
        f.write(hdr)
        f.write(mbytes)
        for r in rows:
            f.write(np.uint64(r).tobytes())
            f.write(codec.encode(plan.row_ops[r]))
    print(f"[overlay] wrote {len(rows)} rows -> {out_path} "
          f"({os.path.getsize(out_path)} bytes)")


def materialize(shards, loc: TableLocation, plan: Plan, out_dir: str,
                out_name: str, dry_run: bool) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    codec = RowCodec(loc.qtype, loc.row_dim)
    # hardlink every shard, then replace the table shard with a patched copy.
    # A dry run must write NOTHING, so skip the whole link/copy phase then.
    if not dry_run:
        for sh in shards:
            dst = os.path.join(out_dir, _shard_out_name(os.path.basename(sh.path), out_name))
            if os.path.abspath(dst) == os.path.abspath(sh.path):
                continue
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(sh.path, dst)  # cheap
                linked = True
            except OSError:
                linked = False
            if sh.shard_no == loc.shard_no and not linked:
                pass  # will copy below
            if sh.shard_no != loc.shard_no and not linked:
                shutil.copy2(sh.path, dst)
    table_out = os.path.join(out_dir, _shard_out_name(os.path.basename(loc.path), out_name))
    if os.path.abspath(table_out) != os.path.abspath(loc.path):
        # need a real (writable) copy of the table shard
        if os.path.exists(table_out) and os.stat(table_out).st_nlink > 1:
            os.remove(table_out)
        if not os.path.exists(table_out):
            if dry_run:
                print(f"[materialize] would copy table shard {os.path.basename(loc.path)} "
                      f"({os.path.getsize(loc.path)/1e9:.1f} GB) -> {table_out}")
            else:
                print(f"[materialize] copying table shard ({os.path.getsize(loc.path)/1e9:.1f} GB) ...")
                shutil.copy2(loc.path, table_out)
    if not dry_run:
        with open(table_out, "r+b") as f:
            for r, vec in sorted(plan.row_ops.items()):
                f.seek(loc.data_offset + int(r) * codec.bpr)
                f.write(codec.encode(vec))
    print(f"[materialize] updated table shard: {table_out}")
    return out_dir


def _shard_out_name(base: str, out_name: str) -> str:
    import re
    m = re.search(r"(\d{5})-of-(\d{5})\.gguf", base)
    if m:
        stem = out_name[:-5] if out_name.endswith(".gguf") else out_name
        return f"{stem}-{m.group(1)}-of-{m.group(2)}.gguf"
    return out_name


def in_place_patch(loc: TableLocation, plan: Plan, undo_path: str, dry_run: bool) -> None:
    codec = RowCodec(loc.qtype, loc.row_dim)
    rows = sorted(plan.row_ops)
    if not dry_run:
        # back up original bytes for undo
        with open(loc.path, "rb") as f, open(undo_path, "wb") as u:
            u.write(b"PLEUNDO1")
            u.write(json.dumps({"path": loc.path, "data_offset": loc.data_offset,
                                "bytes_per_row": codec.bpr,
                                "rows": [int(r) for r in rows]}).encode() + b"\n")
            for r in rows:
                f.seek(loc.data_offset + int(r) * codec.bpr)
                u.write(f.read(codec.bpr))
        with open(loc.path, "r+b") as f:
            for r in rows:
                f.seek(loc.data_offset + int(r) * codec.bpr)
                f.write(codec.encode(plan.row_ops[r]))
    print(f"[in-place] patched {len(rows)} rows in {loc.path}; undo -> {undo_path}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="path to any shard of the model")
    ap.add_argument("--knowledge", required=True, help="knowledge JSON file")
    ap.add_argument("--mode", choices=["materialize", "overlay", "in-place"],
                    default="overlay")
    ap.add_argument("--out", default=None, help="output dir (materialize) or file (overlay)")
    ap.add_argument("--tokenizer-json", default=None, help="HF tokenizer.json (optional)")
    ap.add_argument("--prefer-tokenizer", choices=["auto", "hf", "gguf"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="plan + collision report only")
    ap.add_argument("--report", default=None, help="write JSON plan report here")
    args = ap.parse_args(argv)

    shards = discover_shards(args.gguf)
    meta = read_metadata(shards[0].path)
    c = load_ple_constants(meta)
    loc = locate_table(shards)
    tok, which = make_tokenizer(shards[0].path, args.tokenizer_json, args.prefer_tokenizer)

    print(f"model={meta.get('general.name')} arch={meta.get('general.architecture')} "
          f"shards={len(shards)}")
    print(f"table={loc.name} rows={loc.n_rows} row_dim={loc.row_dim} qtype={loc.qtype.name} "
          f"bytes/row={bytes_per_row(loc.qtype, loc.row_dim)} in shard {loc.shard_no}")
    print(f"ple: ngram_size={c.ngram_size} heads/ngram={c.heads_per_ngram} "
          f"n_heads={c.n_heads} eos={c.eos_token_id} tokenizer={which}")

    entries, _ = load_knowledge(args.knowledge)
    base_dir = os.path.dirname(os.path.abspath(args.knowledge))
    plan = build_plan(entries, tok, c, loc, base_dir)

    print(f"\nplan: {len(plan.row_ops)} unique rows to write "
          f"({len(plan.touched)} touched by entries)")
    shared = {r: n for r, n in plan.touched.items() if n > 1}
    if shared:
        print(f"note: {len(shared)} rows touched by >1 entry (hash collisions merge them)")
    for rep in plan.report:
        print("  -", json.dumps(rep, ensure_ascii=False))

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"rows": len(plan.row_ops), "report": plan.report,
                       "table_sha1_before": sha1_rows(loc, plan.row_ops)
                       if plan.row_ops else None}, f, indent=2)

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    if args.mode == "overlay":
        out = args.out or os.path.splitext(args.knowledge)[0] + ".plepatch"
        write_overlay(out, plan, loc, c, entries, str(meta.get("general.name")))
    elif args.mode == "materialize":
        out_dir = args.out or (os.path.splitext(args.gguf)[0] + ".injected")
        out_name = os.path.basename(out_dir) + ".gguf" if not out_dir.endswith(".gguf") else os.path.basename(out_dir)
        materialize(shards, loc, plan, os.path.dirname(os.path.abspath(out_dir)) or ".",
                    os.path.basename(out_name), args.dry_run)
    elif args.mode == "in-place":
        undo = args.out or (loc.path + ".undo")
        in_place_patch(loc, plan, undo, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
