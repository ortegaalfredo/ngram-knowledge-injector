#!/usr/bin/env python3
"""Inspect the Qwen3.8-Flash-Next (qwen4exp) PLE n-gram table: dump row vectors
and label them with plain-English probe contexts.

What is honestly possible
-------------------------
The PLE table lives in a 160-d per-layer-input space. No LM head is tied to it,
so there is no exact vector -> text decoder. Two sound ways to see "which
concepts are in the PLE":

  1. EXACT collisions  -- the runtime hashes a context into n_heads rows; any
     probe context that hashes to row R *shares that row at inference*. This is
     ground truth, not a heuristic.
  2. Cosine labeling   -- rows carry learned content; comparing a row's content
     against the rows of labeled probe contexts surfaces which probes behave
     like it. Correlative evidence, printed with its cosine so you can judge.

Subcommands
-----------
  context  show the rows a context maps to (what the model reads for it)
  rows     dump raw row vectors (explicit ids or a range) to text/.npz/.npy
  probes   auto-generate a probe list from the tokenizer vocabulary
  decode   label one row (or a context's rows) against a probe set
  scan     sample random rows and label each -> a browseable map of the table
  stats    global table statistics from a random sample (norms, dead rows)

All reads are sparse (fseek per row); nothing here loads the 54 GB table.

Examples:
  python3 ple_dump.py context --gguf model/...-00001-of-00006.gguf \
      --text "The capital of France is"
  python3 ple_dump.py probes --gguf model/...gguf --words 300 --pairs 1500 \
      --out ngt/probes.txt
  python3 ple_dump.py decode --gguf model/...gguf --row 11681901 \
      --probes ngt/probes.txt -k 8
  python3 ple_dump.py scan  --gguf model/...gguf --probes ngt/probes.txt \
      --samples 200 --seed 0
  python3 ple_dump.py rows  --gguf model/...gguf --range 0:1000 --out ngt/head.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ple_core import (  # noqa: E402
    PleConstants, TableLocation, discover_shards, load_ple_constants,
    locate_table, read_metadata, read_rows, rows_for_sequence,
)
from ple_tok import make_tokenizer  # noqa: E402


# --------------------------------------------------------------------------- #
# shared loading
# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self, gguf: str, tokenizer_json: str | None):
        self.shards = discover_shards(gguf)
        meta = read_metadata(self.shards[0].path)
        self.c = load_ple_constants(meta)
        self.loc = locate_table(self.shards)
        self.tok, self.tok_kind = make_tokenizer(self.shards[0].path,
                                                 tokenizer_json, "auto")

    def rows_for(self, text: str) -> list[list[int]]:
        ids = self.tok.encode(text)
        if not ids:
            raise ValueError(f"text tokenized to zero tokens: {text!r}")
        return ids, rows_for_sequence(ids, self.c)


def row_stats(v: np.ndarray) -> tuple[float, float, float]:
    n = float(np.linalg.norm(v))
    return n, float(v.mean()), float(v.std())


# --------------------------------------------------------------------------- #
# probe set: labeled contexts -> (label, head, row_id, vector)
# --------------------------------------------------------------------------- #
def build_probe_vectors(ctx: Ctx, probe_lines: list[str],
                        progress: bool = False):
    labels: list[str] = []
    heads: list[int] = []
    row_ids: list[int] = []
    vecs: list[np.ndarray] = []
    need = set()
    per_probe = []
    for text in probe_lines:
        text = text.strip()
        if not text or text.startswith("#"):
            continue
        try:
            _, all_rows = ctx.rows_for(text)
        except Exception:
            continue
        last = all_rows[-1]
        start = len(labels)
        for h, r in enumerate(last):
            labels.append(text)
            heads.append(h)
            row_ids.append(int(r))
            need.add(int(r))
        per_probe.append((text, start, len(labels)))
    if not need:
        raise SystemExit("no usable probes")
    mat = read_rows(ctx.loc, sorted(need))
    index = {r: i for i, r in enumerate(sorted(need))}
    for k, r in enumerate(row_ids):
        vecs.append(mat[index[r]])
    V = np.asarray(vecs, dtype=np.float32)          # (P*H, row_dim)
    n = np.linalg.norm(V, axis=1)
    n[n == 0] = 1.0
    Vn = V / n[:, None]
    if progress:
        print(f"probes: {len(per_probe)} contexts, {V.shape[0]} head-vectors, "
              f"{len(need)} distinct rows", file=sys.stderr)
    return {"labels": np.array(labels), "heads": np.array(heads),
            "row_ids": np.array(row_ids, dtype=np.uint64),
            "V": V, "Vn": Vn, "per_probe": per_probe}


def load_probes(ctx: Ctx, path: str, cache: str | None, progress: bool):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        V = z["V"].astype(np.float32)
        n = np.linalg.norm(V, axis=1)
        n[n == 0] = 1.0
        return {"labels": z["labels"], "heads": z["heads"],
                "row_ids": z["row_ids"], "V": V, "Vn": V / n[:, None],
                "per_probe": []}
    with open(path, "r", encoding="utf-8") as f:
        p = build_probe_vectors(ctx, f.readlines(), progress=progress)
    if cache:
        np.savez_compressed(cache, labels=p["labels"], heads=p["heads"],
                            row_ids=p["row_ids"], V=p["V"])
    return p


def label_row(pv: dict, vec: np.ndarray, k: int,
              exclude_probe_of: str | None = None) -> list[tuple[float, str, int, int]]:
    n = np.linalg.norm(vec)
    q = vec / (n if n else 1.0)
    cos = pv["Vn"] @ q
    order = np.argsort(-cos)
    out, seen_texts = [], set()
    for i in order:
        lab = str(pv["labels"][i])
        if exclude_probe_of is not None and lab == exclude_probe_of:
            continue
        if lab in seen_texts:
            continue
        seen_texts.add(lab)
        out.append((float(cos[i]), lab, int(pv["heads"][i]),
                    int(pv["row_ids"][i])))
        if len(out) >= k:
            break
    return out


def fmt_neighbors(neigh, cos_min: float) -> str:
    if not neigh:
        return "  (no probes)"
    parts = []
    for cos, lab, h, rid in neigh:
        tag = "*" if cos < cos_min else " "
        parts.append(f"{tag}{cos:.3f} {lab!r}(h{h})")
    return "  " + " | ".join(parts)


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_context(ctx: Ctx, a) -> int:
    ids, all_rows = ctx.rows_for(a.text)
    print(f"text: {a.text!r}")
    print(f"tokens ({len(ids)}): {ids}")
    try:
        dec = [ctx.tok.decode([i]) for i in ids]
        print(f"pieces: {[repr(d) for d in dec]}")
    except Exception:
        pass
    print(f"n_heads={ctx.c.n_heads} (heads 0-{ctx.c.heads_per_ngram - 1}=bigram, "
          f"{ctx.c.heads_per_ngram}-{ctx.c.n_heads - 1}=trigram)")
    pos_list = (range(len(ids)) if a.at == "all"
                else [len(ids) - 1] if a.at == "last" else [int(a.at)])
    rows = sorted({r for p in pos_list for r in all_rows[p]})
    mat = read_rows(ctx.loc, rows)
    idx = {r: i for i, r in enumerate(rows)}
    for p in pos_list:
        print(f"position {p}:")
        for h, r in enumerate(all_rows[p]):
            v = mat[idx[r]]
            n, m, s = row_stats(v)
            print(f"  head {h:2d} -> row {r:<10d} norm={n:.4f} mean={m:+.5f} "
                  f"std={s:.4f}")
    if a.out:
        np.save(a.out, mat)
        print(f"saved {len(rows)} row vectors -> {a.out} "
              f"(shape {mat.shape}, float32)")
    return 0


def cmd_rows(ctx: Ctx, a) -> int:
    if a.range:
        start_s, count_s = a.range.split(":")
        start, count = int(start_s), int(count_s)
        if start < 0 or count <= 0 or start + count > ctx.loc.n_rows:
            raise SystemExit(f"range out of bounds 0..{ctx.loc.n_rows}")
        ids = list(range(start, start + count))
    elif a.rows:
        ids = [int(x) for x in a.rows.split(",")]
    else:
        raise SystemExit("need --rows or --range")
    bad = [r for r in ids if not 0 <= r < ctx.loc.n_rows]
    if bad:
        raise SystemExit(f"row ids out of bounds: {bad[:5]}")
    mat = read_rows(ctx.loc, ids)
    if a.out:
        np.save(a.out, mat)
        np.save(a.out.replace(".npy", ".ids.npy"), np.array(ids, dtype=np.uint64))
        print(f"saved {len(ids)} rows -> {a.out} (float32, row-major, "
              f"ids in {os.path.splitext(a.out)[0]}.ids.npy)")
    if a.raw:
        for i, r in enumerate(ids):
            s = np.array2string(mat[i], precision=4, max_line_width=200,
                                threshold=16)
            print(f"row {r}: {s}")
    else:
        for i, r in enumerate(ids):
            n, m, s = row_stats(mat[i])
            print(f"row {r:<10d} norm={n:.4f} mean={m:+.5f} std={s:.4f} "
                  f"absmax={np.abs(mat[i]).max():.4f}")
    return 0


def cmd_probes(ctx: Ctx, a) -> int:
    vocab = getattr(ctx.tok, "tokens", None)
    if not vocab:
        raise SystemExit("tokenizer does not expose a vocabulary")
    words = []
    seen = set()
    for piece in vocab:
        w = str(piece).replace("\u0120", " ").strip()
        if len(w) < 3 or not w.isalpha() or not w.isascii():
            continue
        if w.startswith("<") and w.endswith(">"):
            continue
        wl = w.lower()
        if wl in seen:
            continue
        seen.add(wl)
        words.append(wl)
        if len(words) >= a.words:
            break
    lines = list(words)
    rng = np.random.default_rng(a.seed)
    if a.pairs and len(words) >= 2:
        i = rng.integers(0, len(words), size=a.pairs)
        j = rng.integers(0, len(words), size=a.pairs)
        for x, y in zip(i, j):
            if x != y:
                lines.append(f"{words[x]} {words[y]}")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# one probe context per line (auto-generated from vocab)\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} probes -> {a.out} "
          f"({len(words)} single words + "
          f"{max(0, len(lines) - len(words))} pairs)")
    return 0


def cmd_decode(ctx: Ctx, a) -> int:
    pv = load_probes(ctx, a.probes, a.probe_cache, progress=True)
    cos_min = a.cos_min
    if a.row is not None:
        targets = [(int(r), None) for r in a.row.split(",")]
    elif a.text:
        _, all_rows = ctx.rows_for(a.text)
        last = all_rows[-1]
        print(f"context {a.text!r} -> last-position rows:")
        targets = [(int(r), a.text) for r in last]
    else:
        raise SystemExit("need --row or --text")
    for row_id, src_text in targets:
        v = read_rows(ctx.loc, [row_id])[0]
        n, m, s = row_stats(v)
        print(f"row {row_id} norm={n:.4f} mean={m:+.5f} std={s:.4f}")
        hits = [(int(h), int(rr)) for h, rr in
                zip(pv["heads"], pv["row_ids"]) if int(rr) == row_id]
        if hits:
            labs = sorted({str(pv["labels"][k]) for k in range(len(pv["row_ids"]))
                           if int(pv["row_ids"][k]) == row_id})
            hs = [h for h, _ in hits]
            print(f"  EXACT: shared by probe contexts {labs} (heads {hs})")
        else:
            print("  EXACT: no probe hashes here (row unprobed)")
        neigh = label_row(pv, v, a.k, exclude_probe_of=src_text)
        print(f"  nearest probe rows (cosine, * below {cos_min}):")
        print(fmt_neighbors(neigh, cos_min) if neigh else "  (none)")
    return 0


def cmd_scan(ctx: Ctx, a) -> int:
    pv = load_probes(ctx, a.probes, a.probe_cache, progress=True)
    rng = np.random.default_rng(a.seed)
    ids = rng.choice(ctx.loc.n_rows, size=a.samples, replace=False)
    ids = sorted(int(x) for x in ids)
    mat = read_rows(ctx.loc, ids)
    print(f"scanning {len(ids)} random rows of {ctx.loc.n_rows} "
          f"against {len(pv['per_probe']) or len(set(pv['labels'].tolist()))} "
          f"probe contexts (top-{a.k} shown, * below {a.cos_min})")
    hits = 0
    for i, r in enumerate(ids):
        v = mat[i]
        n = float(np.linalg.norm(v))
        neigh = label_row(pv, v, a.k)
        best = neigh[0][0] if neigh else 0.0
        margin = (neigh[0][0] - neigh[1][0]) if len(neigh) > 1 else best
        exact = [str(pv["labels"][k]) for k in
                 range(len(pv["row_ids"])) if int(pv["row_ids"][k]) == r]
        if exact or best >= a.cos_min:
            hits += 1
            tag = f"EXACT {exact}" if exact else f"cos={best:.3f} (margin {margin:.3f})"
            print(f"row {r:<10d} norm={n:.4f}  {tag}")
            print(fmt_neighbors(neigh, a.cos_min))
    print(f"\n{hits}/{len(ids)} rows labeled above cos {a.cos_min} "
          + (f"; the remaining {len(ids) - hits} carry content the probe set "
             f"does not name (expected: probes cover a sliver of the table)."
             if hits < len(ids) else "."))
    return 0


def cmd_stats(ctx: Ctx, a) -> int:
    rng = np.random.default_rng(a.seed)
    ids = sorted(int(x) for x in
                 rng.choice(ctx.loc.n_rows, size=a.samples, replace=False))
    mat = read_rows(ctx.loc, ids)
    norms = np.linalg.norm(mat, axis=1)
    dead = int((norms < 1e-4).sum())
    print(f"table  : {ctx.loc.name}  rows={ctx.loc.n_rows} "
          f"row_dim={ctx.loc.row_dim} qtype={ctx.loc.qtype.name} "
          f"shard={ctx.loc.shard_no}")
    print(f"sample : {len(ids)} random rows")
    print(f"norm   : min={norms.min():.5f} p25={np.percentile(norms, 25):.5f} "
          f"med={np.median(norms):.5f} p75={np.percentile(norms, 75):.5f} "
          f"max={norms.max():.5f}")
    print(f"dead rows (norm < 1e-4): {dead} ({100.0 * dead / len(ids):.2f}%)")
    print(f"interpretation: norms ~{np.median(norms):.3f} are the signal "
          f"magnitude added to the residual stream when a context hashes here")
    if a.out:
        np.save(a.out, mat)
        print(f"sample matrix saved -> {a.out}")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="path to any shard of the model")
    ap.add_argument("--tokenizer-json", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("context", help="rows a context maps to")
    p.add_argument("--text", required=True)
    p.add_argument("--at", default="last", help="last|all|<position>")
    p.add_argument("--out", help="save row vectors (.npy)")
    p.set_defaults(fn=cmd_context)

    p = sub.add_parser("rows", help="dump raw row vectors")
    p.add_argument("--rows", help="comma list of row ids")
    p.add_argument("--range", help="START:COUNT contiguous rows")
    p.add_argument("--out", help="save as .npy (float32)")
    p.add_argument("--raw", action="store_true", help="print full vectors")
    p.set_defaults(fn=cmd_rows)

    p = sub.add_parser("probes", help="auto-generate probe list from vocab")
    p.add_argument("--words", type=int, default=300)
    p.add_argument("--pairs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_probes)

    p = sub.add_parser("decode", help="label a row/context against probes")
    p.add_argument("--row", help="comma list of row ids")
    p.add_argument("--text", help="label the rows of this context")
    p.add_argument("--probes", required=True, help="probe list (one context/line)")
    p.add_argument("--probe-cache", help=".npz cache for probe vectors")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--cos-min", type=float, default=0.25,
                   help="cosine below this is reported as weak")
    p.set_defaults(fn=cmd_decode)

    p = sub.add_parser("scan", help="sample rows and label each")
    p.add_argument("--probes", required=True)
    p.add_argument("--probe-cache", help=".npz cache for probe vectors")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--cos-min", type=float, default=0.35)
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("stats", help="global table statistics")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="save the sample matrix (.npy)")
    p.set_defaults(fn=cmd_stats)

    a = ap.parse_args(argv)
    ctx = Ctx(a.gguf, a.tokenizer_json)
    return a.fn(ctx, a)


if __name__ == "__main__":
    raise SystemExit(main())
