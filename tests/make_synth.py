"""Build a tiny synthetic qwen4exp GGUF with a real-size (sparse) PLE table so
the test suite runs without the 188 GB Qwen3.8-Flash-Next weights.

Hash constants (ngram_size, heads_per_ngram, layer_multipliers, per-head
vocab sizes/offsets, eos/image ids) are the REAL ones taken from the actual
model -- identical to tests/golden/ref2.cpp -- so test_hash_matches_cpp_golden
validates the same math with or without the real weights.

The per_layer_token_embd.weight tensor is declared with the full
row count (sum of head vocab sizes ~= 320M rows, ~54 GB as Q8_0) but only the
first N_MATERIALIZE rows are actually written: the GGUF header is patched in
place and the file is extended as a SPARSE file, so it occupies a few MB of
disk while reads from untouched rows return zeros (valid Q8_0 blocks).

Usage: python3 tests/make_synth.py [-o build/synth-qwen4exp.gguf]
"""
import argparse
import os
import struct

import numpy as np
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType as Q, GGUFValueType
from gguf.quants import quantize

ARCH = "qwen4exp"
ROW_DIM = 160
N_MATERIALIZE = 16384          # rows actually written (~2.8 MB as Q8_0)
TENSOR = "per_layer_token_embd.weight"
EOS_TOKEN_ID = 248044
IMAGE_TOKEN_ID = 248056
# real values (source of truth: tests/golden/ref2.cpp, which was generated
# from the actual Qwen3.8-Flash-Next-Q8_0 metadata)
MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
HEAD_VOCAB = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
              20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
              20000153, 20000159, 20000161, 20000171]
N_ROWS = sum(HEAD_VOCAB)       # 320,001,446
BPR = ROW_DIM // 32 * 34       # Q8_0 bytes per row (160)


def bytes_to_unicode() -> dict:
    """Standard GPT-2 reversible byte <-> unicode-char table."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def build_vocab():
    b2u = bytes_to_unicode()
    toks = [b2u[b] for b in range(256)]          # 1 token per byte, no merges
    toks += ["hello", "world", "France", "capital", "red"]  # sample words
    return toks


def build(out_path: str, seed: int = 0) -> str:
    offs, o = [], 0
    for v in HEAD_VOCAB:
        offs.append(o)
        o += v
    assert o == N_ROWS

    rng = np.random.default_rng(seed)
    rows = rng.normal(0.0, 0.01, size=(N_MATERIALIZE, ROW_DIM)).astype(np.float32)
    rows[0] = 0.0                                 # keep row 0 all-zero
    packed = np.ascontiguousarray(
        quantize(rows, Q.Q8_0).reshape(N_MATERIALIZE, ROW_DIM // 32, 34),
        dtype=np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w = GGUFWriter(out_path, ARCH)
    w.add_architecture()
    w.add_name("Qwen3.8 Flash Next (synthetic)")
    w.add_context_length(1024)
    w.add_block_count(2)
    # PLE constants -- the real ones (ple_core.load_ple_constants reads these)
    w.add_key_value(f"{ARCH}.ple.layers", [1], GGUFValueType.ARRAY,
                    sub_type=GGUFValueType.UINT32)
    w.add_uint32(f"{ARCH}.embedding_length_per_layer_input", ROW_DIM)
    w.add_uint32(f"{ARCH}.ple.ngram_size", 3)
    w.add_uint32(f"{ARCH}.ple.heads_per_ngram", 8)
    w.add_uint32(f"{ARCH}.ple.conv_kernel", 4)
    w.add_uint32(f"{ARCH}.ple.eos_token_id", EOS_TOKEN_ID)
    w.add_uint32(f"{ARCH}.ple.image_token_id", IMAGE_TOKEN_ID)
    w.add_key_value(f"{ARCH}.ple.layer_multipliers", [int(x) for x in MULTIPLIERS],
                    GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
    w.add_key_value(f"{ARCH}.ple.head_offsets", [int(x) for x in offs],
                    GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
    w.add_key_value(f"{ARCH}.ple.head_vocab_sizes", [int(x) for x in HEAD_VOCAB],
                    GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
    # minimal GPT-2 byte-level tokenizer (any string round-trips)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt2")
    w.add_token_list(build_vocab())
    w.add_token_merges([])
    w.add_eos_token_id(EOS_TOKEN_ID)
    w.add_unk_token_id(0)
    # the PLE table: written with N_MATERIALIZE rows, header patched below
    w.add_tensor(TENSOR, packed,
                 raw_shape=np.array([N_MATERIALIZE, BPR], dtype=np.uint32),
                 raw_dtype=Q.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    _patch_and_extend(out_path)
    print(f"built {out_path}")
    print(f"  table: {ROW_DIM} x {N_ROWS:,} rows declared "
          f"({N_ROWS * BPR / 1e9:.1f} GB apparent, "
          f"{os.path.getsize(out_path) / 1e6:.1f} MB on disk sparse)")
    return out_path


def _patch_and_extend(path: str) -> None:
    """Grow the declared tensor to N_ROWS and extend the file sparsely.

    Tensor-info layout in GGUF v3: name (vlen str), n_dims u32, dims u64[n],
    type u32, offset u64. Only dims[1] changes value; the on-disk size is then
    grown with a hole so reads of untouched rows return zeros. (GGUFReader is
    deliberately NOT used here: it eagerly maps the whole tensor and would
    fail on the size mismatch before we get a chance to extend the file.)
    """
    name = TENSOR.encode()
    with open(path, "r+b") as f:
        buf = bytearray(f.read())
        i = buf.find(name)
        assert i != -1, "tensor name not found in header"
        p = i + len(name)
        (ndims,) = struct.unpack_from("<I", buf, p)
        assert ndims == 2, f"expected 2D tensor, got {ndims}"
        d0, d1 = struct.unpack_from("<QQ", buf, p + 4)
        assert d0 == ROW_DIM and d1 == N_MATERIALIZE, (d0, d1)
        struct.pack_into("<Q", buf, p + 4 + 8, N_ROWS)
        f.seek(0)
        f.write(buf)
        # the written tensor data ends at EOF; extend with a hole to N_ROWS
        end = f.tell()
    os.truncate(path, end + (N_ROWS - N_MATERIALIZE) * BPR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=os.path.join("build", "synth-qwen4exp.gguf"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.out, args.seed)
