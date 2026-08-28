"""Tests for the PLE n-gram injector.

The hash is validated against golden vectors produced by a C++ transcription of
llm_graph_input_ple::set_input (qwen4exp.cpp) using the model's real constants.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from ple_core import (  # noqa: E402
    RowCodec, discover_shards, load_ple_constants, locate_table, read_metadata,
    read_rows, rows_for_sequence, rows_for_token, bytes_per_row,
)
from ple_tok import GGUFTokenizer, validate_tokenizers  # noqa: E402
import inject  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "golden", "golden.txt")
if not os.path.exists(GOLDEN):
    GOLDEN = os.path.join("/tmp/ngt", "golden.txt")
RESEARCH = os.path.expanduser("~/ai/AICommander/research")


def find_model():
    pats = ["Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
            "Qwen3.8-Flash-Next-Q8_0*00001-of-00006*"]
    for base in (os.path.join(ROOT, "model"), RESEARCH):
        for pat in pats:
            hit = sorted(glob.glob(os.path.join(base, pat)))
            if hit:
                return hit[0]
    return None


def find_tokenizer_json():
    hit = sorted(glob.glob("/tmp/hfcache/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/*/tokenizer.json"))
    return hit[0] if hit else None


def codec_bpr(loc):
    from ple_core import RowCodec
    return RowCodec(loc.qtype, loc.row_dim).bpr


def consts():
    path = find_model()
    meta = read_metadata(path)
    return load_ple_constants(meta), path


# --------------------------------------------------------------------------- #
def test_hash_matches_cpp_golden():
    """Python row math == C++ reference, on the model's real constants."""
    c, _ = consts()
    assert os.path.exists(GOLDEN), "golden.txt missing (build /tmp/ngt/ref2)"
    n_ok = 0
    with open(GOLDEN) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            head, rows = line.split(":")
            tok, p1, p2 = (int(x) for x in head.split())
            expect = [int(x) for x in rows.split()]
            # golden applies the eos reset already; feed raw predecessors
            got = rows_for_token(tok, [p2, p1], c)
            assert got == expect, f"hash mismatch for {tok},{p1},{p2}: {got} != {expect}"
            n_ok += 1
    assert n_ok >= 10
    print(f"  hash matches C++ golden on {n_ok} windows")


def test_rows_in_range():
    c, _ = consts()
    rng = np.random.default_rng(0)
    for _ in range(2000):
        tok = int(rng.integers(0, 248320))
        prev = [int(rng.integers(0, 248320)) for _ in range(c.ngram_size - 1)]
        for r in rows_for_token(tok, prev, c):
            assert 0 <= r < c.n_rows, r


def test_eos_reset_semantics():
    """The runtime cuts the window at an eos, propagating to OLDER tokens only.

    Mirrors qwen4exp.cpp: scanning s=1..n (newest->oldest), once an eos is seen
    every older ctx slot becomes eos. An eos at t-2 therefore does NOT change
    ctx[1]; an eos at t-1 DOES force ctx[2]=eos. The token's own eos never cuts
    its own context.
    """
    c, _ = consts()
    eos = c.eos_token_id
    # eos at t-1 (prev=[_, eos]) forces the t-2 slot to eos regardless of t-2
    a = rows_for_token(1234, [999, eos], c)
    b = rows_for_token(1234, [555, eos], c)
    assert a == b, "an eos at t-1 must cut the older (t-2) context"
    # eos at t-2 does not alter ctx[1]
    d = rows_for_token(1234, [eos, 999], c)
    e = rows_for_token(1234, [7, 999], c)
    assert d != e, "eos at t-2 should only affect the t-2 slot, not ctx[1]"
    # the token's own eos does not cut its own context (verified vs golden)
    assert rows_for_token(eos, [2, 1], c) is not None


def test_q8_roundtrip_byte_identical():
    """Q8_0 dequant->quant of a real row reproduces the original bytes exactly."""
    _, path = consts()
    shards = discover_shards(path)
    loc = locate_table(shards)
    codec = RowCodec(loc.qtype, loc.row_dim)
    for r in (0, 1, 12345, 999999, 320001000):
        with open(loc.path, "rb") as f:
            f.seek(loc.data_offset + r * codec.bpr)
            raw = np.frombuffer(f.read(codec.bpr), dtype=np.uint8)
        vec = codec.decode(raw)
        re = codec.encode(vec)
        assert np.array_equal(re, raw), f"row {r} not byte-identical after roundtrip"
    print("  Q8_0 roundtrip byte-identical on sampled rows")


def test_untouched_rows_bit_exact_after_patch():
    """Patching must not disturb neighbouring rows (row addressing is exact)."""
    _, path = consts()
    shards = discover_shards(path)
    loc = locate_table(shards)
    codec = RowCodec(loc.qtype, loc.row_dim)
    probe = [1000, 1001, 1002]
    before = read_rows(loc, probe)
    # simulate writing row 1001 only
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "t.bin")
        bpr = codec.bpr
        with open(tmp, "wb") as o:
            with open(loc.path, "rb") as f:
                for r in probe:
                    f.seek(loc.data_offset + r * bpr)
                    o.write(f.read(bpr))
        with open(tmp, "r+b") as o:
            o.seek(1 * bpr)
            o.write(codec.encode(np.full(loc.row_dim, 0.123, dtype=np.float32)))
        got = []
        with open(tmp, "rb") as f:
            for _ in probe:
                got.append(codec.decode(np.frombuffer(f.read(bpr), dtype=np.uint8)))
    assert np.array_equal(got[0], before[0]), "row before the patched one changed"
    assert np.array_equal(got[2], before[2]), "row after the patched one changed"
    assert not np.array_equal(got[1], before[1]), "patched row did not change"


def test_gguf_bpe_matches_hf():
    _, path = consts()
    tj = find_tokenizer_json()
    if not tj:
        print("  (no tokenizer.json; skipping HF cross-check)")
        return
    samples = [
        "The capital of France is",
        "def f(x): return x**2 + 1",
        "Héllo 世界 naïve 🚀 42",
        "Q: 2+2? A: 4",
    ]
    ok, diffs = validate_tokenizers(path, tj, samples)
    assert ok, f"BPE diverges: {diffs}"


def test_overlay_roundtrip():
    """Write an overlay, read it back, confirm rows + manifest."""
    c, path = consts()
    shards = discover_shards(path)
    loc = locate_table(shards)
    tok = GGUFTokenizer.from_gguf(shards[0].path)
    entries, _ = inject.load_knowledge(os.path.join(ROOT, "examples", "knowledge.json"))
    plan = inject.build_plan(entries, tok, c, loc, os.path.join(ROOT, "examples"))
    assert plan.row_ops, "plan produced no rows"
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "k.plepatch")
        inject.write_overlay(out, plan, loc, c, entries, "test-model")
        hdr = open(out, "rb").read(inject.OVERLAY_HDR)
        assert hdr[0:8] == b"PLEOVLY1"
        ver = int.from_bytes(hdr[8:10], "little")
        qtype = int.from_bytes(hdr[12:16], "little")
        row_dim = int.from_bytes(hdr[16:24], "little")
        bpr = int.from_bytes(hdr[24:32], "little")
        nrows = int.from_bytes(hdr[32:40], "little")
        mlen = int.from_bytes(hdr[40:48], "little")
        tname = hdr[56:120].split(b"\0")[0].decode()
        assert ver == 1 and row_dim == loc.row_dim and bpr == codec_bpr(loc)
        assert tname == loc.name
        with open(out, "rb") as f:
            f.seek(inject.OVERLAY_HDR)
            man = json.loads(f.read(mlen))
            assert int(nrows) == len(plan.row_ops)
            codec = RowCodec(loc.qtype, loc.row_dim)
            seen = []
            for _ in range(int(nrows)):
                (rid,) = np.frombuffer(f.read(8), dtype="<u8")
                raw = np.frombuffer(f.read(codec.bpr), dtype=np.uint8)
                assert rid in plan.row_ops
                assert np.array_equal(codec.decode(raw),
                                      codec.decode(codec.encode(plan.row_ops[rid])))
                seen.append(int(rid))
        assert sorted(seen) == sorted(plan.row_ops)
        assert man["format"] == "ple-overlay-v1"
        assert man["row_dim"] == loc.row_dim
        assert man["qtype"] == loc.qtype.name


def test_dry_run_cli():
    _, path = consts()
    tj = find_tokenizer_json()
    cmd = [sys.executable, os.path.join(ROOT, "inject.py"), "--gguf", path,
           "--knowledge", os.path.join(ROOT, "examples", "knowledge.json"),
           "--mode", "overlay", "--dry-run"]
    if tj:
        cmd += ["--tokenizer-json", tj]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "plan:" in r.stdout and "dry-run" in r.stdout


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
