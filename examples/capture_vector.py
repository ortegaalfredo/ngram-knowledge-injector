#!/usr/bin/env python3
"""Capture a concept-direction vector from the PLE n-gram table.

The table is the model's own memory: the runtime hashes a context into
n_heads rows and adds the gathered vectors into the residual stream. Reading
those rows for a concept-rich context therefore yields a vector that is
*exactly* in the per-layer-input space the table lives in (160-d here) — a
ready-to-inject concept direction without any GPU activation hooks.

Examples:
  # capture the direction for "red" from one rich context
  python3 examples/capture_vector.py \
      --gguf model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
      --text "My favorite color is red. Red is the color of" \
      --out examples/red.vec.npy

  # capture from several contexts and average them into one direction
  python3 examples/capture_vector.py --gguf <shard> \
      --text "The red apple is" --text "Red is the color of" \
      --out examples/red.vec.npy

  # merge previously captured vectors
  python3 examples/capture_vector.py --mean red1.vec.npy red2.vec.npy \
      --out examples/red.vec.npy

Then reference it from knowledge.json:
  { "trigger": "Q: What is your favorite color? A:",
    "vector": "red.vec.npy", "op": "add", "scale": 1.0 }

See examples/README.md Part 4 for the full recipe.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from ple_core import (  # noqa: E402
    discover_shards, load_ple_constants, locate_table, read_metadata,
    read_rows, rows_for_sequence,
)
from ple_tok import make_tokenizer  # noqa: E402


def _ctx_mean_vec(texts: list[str], tok, c, loc, at: str,
                  hidx: list[int], label: str) -> np.ndarray:
    """Mean row vector over the given contexts (raw, NOT normalized)."""
    vecs: list[np.ndarray] = []
    for text in texts:
        ids = tok.encode(text)
        if not ids:
            raise ValueError(f"{label} tokenized to zero tokens: {text!r}")
        all_rows = rows_for_sequence(ids, c)
        if at == "all":
            pos = range(len(ids))
        elif at == "last":
            pos = [len(ids) - 1]
        else:
            pos = [int(at)]
        rows = sorted({all_rows[p][h] for p in pos for h in hidx})
        vals = read_rows(loc, rows)                      # (k, row_dim) f32
        v = vals.mean(axis=0).astype(np.float32)         # heads are parallel views
        vecs.append(v)
        print(f"  captured {len(rows):3d} rows from {len(ids):3d} tokens ({label}): "
              f"{text[:60]!r}")
    return np.mean(vecs, axis=0).astype(np.float32)


def capture(texts: list[str], gguf: str, at: str,
            heads: list[int] | None, normalize_each: bool,
            contrast: list[str] | None = None) -> np.ndarray:
    shards = discover_shards(gguf)
    c = load_ple_constants(read_metadata(shards[0].path))
    loc = locate_table(shards)
    tok, which = make_tokenizer(shards[0].path, None, "auto")
    print(f"table={loc.name} rows={loc.n_rows} row_dim={loc.row_dim} "
          f"qtype={loc.qtype.name} in shard {loc.shard_no} tokenizer={which}")
    print(f"ple: ngram={c.ngram_size} heads/ngram={c.heads_per_ngram} "
          f"n_heads={c.n_heads}")

    hidx = heads if heads is not None else list(range(c.n_heads))

    if contrast:
        # contrastive direction: mean(concept rows) - mean(contrast rows).
        # Both come from the table itself (offline, in-distribution); the
        # subtraction cancels the shared skeleton and leaves the concept.
        pos_vec = _ctx_mean_vec(texts, tok, c, loc, at, hidx, "concept")
        neg_vec = _ctx_mean_vec(contrast, tok, c, loc, at, hidx, "contrast")
        out = (pos_vec - neg_vec).astype(np.float32)
        print(f"  contrast: |pos|={np.linalg.norm(pos_vec):.4f} "
              f"|neg|={np.linalg.norm(neg_vec):.4f} "
              f"|diff|={np.linalg.norm(out):.4f}")
        # fall through to the shared normalize below
    else:
        out = _ctx_mean_vec(texts, tok, c, loc, at, hidx, "concept")

    n = np.linalg.norm(out)
    if n > 0:
        out = out / n
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="path to any shard of the model (capture mode)")
    ap.add_argument("--text", action="append", default=[],
                    help="concept context; repeat for multi-context capture")
    ap.add_argument("--at", default="last",
                    help="last|all|<index> — position whose rows to read (default last)")
    ap.add_argument("--heads", default=None,
                    help="comma list of head indices (default: all 16)")
    ap.add_argument("--contrast", action="append", default=None,
                    help="contrastive capture: subtract the mean row vector of "
                         "these contexts from the --text contexts (repeatable). "
                         "Isolates the concept by cancelling the shared skeleton "
                         "— e.g. --text '... is red' --contrast '... is blue'.")
    ap.add_argument("--mean", nargs="+", metavar="VEC",
                    help="merge existing .npy vectors instead of capturing")
    ap.add_argument("--target-norm", type=float, default=None,
                    help="rescale output to this exact L2 norm "
                         "(natural PLE rows are ~0.1; sweep 0.25-4.0)")
    ap.add_argument("--sweep-norms", nargs="+", type=float, default=None,
                    metavar="NORM",
                    help="dose sweep: write one .npy per norm, named "
                         "<out-stem>.n<NORM>.npy (e.g. red.vec.n025.npy). "
                         "Natural PLE rows are ~0.1; norm-40 is destructive. "
                         "Implies --target-norm per dose; --out is the stem.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply output by this factor "
                         "(applied after capture/mean, before norm)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip L2 normalization of the final vector")
    ap.add_argument("--out", required=True, help="output .npy (float32, row_dim long)")
    args = ap.parse_args()

    if args.mean:
        arrs = [np.load(p).astype(np.float32).reshape(-1) for p in args.mean]
        if len({a.size for a in arrs}) != 1:
            raise SystemExit(f"vector lengths differ: {[a.size for a in arrs]}")
        vec = np.mean(arrs, axis=0).astype(np.float32)
    else:
        if not args.gguf or not args.text:
            ap.error("capture mode needs --gguf and at least one --text")
        heads = ([int(h) for h in args.heads.split(",")]
                 if args.heads else None)
        vec = capture(args.text, args.gguf, args.at, heads,
                      normalize_each=False, contrast=args.contrast)

    vec = vec.astype(np.float32)
    if args.scale != 1.0:
        vec = vec * np.float32(args.scale)

    def _rescale(v: np.ndarray, target: float) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v * np.float32(target / n)) if n > 0 else v

    if args.sweep_norms:
        # dose sweep: one file per norm, named <stem>.n<NORM>.npy
        base, _ = os.path.splitext(args.out)
        for target in args.sweep_norms:
            tag = ("%g" % target).replace(".", "")
            out = f"{base}.n{tag}.npy"
            np.save(out, _rescale(vec, target).astype(np.float32))
            print(f"wrote {out}: dim={vec.size} norm={np.linalg.norm(np.load(out)):.4f}")
        return 0

    if args.target_norm is not None:
        vec = _rescale(vec, args.target_norm)
    elif not args.no_normalize:
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
    np.save(args.out, vec)
    print(f"wrote {args.out}: dim={vec.size} norm={np.linalg.norm(vec):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
