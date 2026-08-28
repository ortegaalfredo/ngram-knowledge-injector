# PLE n-gram knowledge injector for Qwen3.8-Flash-Next

Inject knowledge into the **PLE n-gram embedding table** of a
`qwen4exp` (Qwen3.8-Flash-Next) GGUF, without rewriting the 54 GB table.

## How the n-gram table works

Qwen3.8-Flash-Next (`model_type: qwen4exp`) has a 51 B-parameter
`per_layer_token_embd.weight` table of shape `[160, 320001536]`. At every
token the runtime hashes the local context into `n_heads = (ngram_size-1) *
heads_per_ngram = 16` rows of this one shared table and adds the gathered
vectors into the residual stream (a learned, hash-addressed memory).

The hash (from `llm_graph_input_ple::set_input`, `src/models/qwen4exp.cpp`):

```
ctx[0] = token, ctx[s] = s-th predecessor (EOS if missing / after an EOS)
for n in 2..ngram_size:
    mixed = (ctx[0]*m[0]) ^ (ctx[1]*m[1]) ^ ... ^ (ctx[n-1]*m[n-1])   # uint64
    for g in 0..heads_per_ngram-1:
        h   = (n-2)*heads_per_ngram + g
        row = mixed % head_vocab_sizes[h] + head_offsets[h]
```

All constants (`ngram_size`, `heads_per_ngram`, `layer_multipliers`,
`head_offsets`, `head_vocab_sizes`, `eos_token_id`) are read from the GGUF
KV metadata, so the tool works for any `qwen4exp` checkpoint.

## Components

| File | Role |
|---|---|
| `ple_core.py` | hash, row addressing, GGUF metadata/shard/table discovery, Q8_0 row codec |
| `ple_tok.py` | tokenizer (HF `tokenizer.json` or GGUF-embedded BPE) |
| `inject.py` | CLI: knowledge JSON -> plan -> overlay / materialize / in-place |
| `ple_dump.py` | table inspector: dump rows, decode rows/contexts against probe contexts |
| `llama_patch/ple-overlay.patch` | llama.cpp patch: COW overlay applied at model load (see `llama_patch/README.md`) |
| `tests/test_ple.py` | 8 tests incl. hash vs C++ golden vectors (`tests/golden/`) |
| `examples/knowledge.json` | example knowledge file |
| `examples/README.md` | full step-by-step guide (edit -> inject -> load -> materialize -> concept vectors -> inspect) |
| `examples/capture_vector.py` | capture a concept-direction vector from the table itself |
| `docs/qwen3-next-ngram-research.md` | research notes: what the PLE table is (and is not) |

## Project layout

```
ngram_tool/
├── ple_core.py            # hash + row addressing + GGUF I/O (no heavy deps)
├── ple_tok.py             # tokenization identical to llama.cpp
├── inject.py              # write knowledge into the table (3 modes)
├── ple_dump.py            # read/inspect the table
├── examples/              # knowledge.json + step-by-step guide + vector capture
├── llama_patch/           # llama.cpp overlay patch + C++ tests + build script
├── tests/                 # python suite + C++ golden hash vectors
├── docs/                  # background research
└── model/                 # put the Qwen3.8-Flash-Next GGUF shards here (gitignored)
```

## Knowledge JSON

```json
{
  "defaults": { "at": "last", "heads": "all", "orders": "all",
                "op": "blend", "alpha": 0.5, "scale": 1.0 },
  "entries": [
    { "trigger": "The capital of France is",
      "vector": "random", "seed": 1, "op": "blend", "alpha": 0.3 },
    { "trigger": "Boiling point of water is",
      "vector": [ /* 160 floats */ ], "op": "set", "orders": [3] },
    { "trigger": "mitochondria is the",
      "copy_from": "powerhouse of the cell", "op": "copy_from" }
  ]
}
```

- `at`: `last` | `first` | `all` | token index — which positions' rows to write.
- `orders`: `all` or list of n-gram orders (`[2]` bigram heads, `[3]` trigram).
- `heads`: `all` or explicit head indices.
- `op`: `blend` (alpha), `set`, `add` (scale), `scale`, `zero`, `copy_from`.
- `vector`: 160 floats, `"random"` (with `seed`), `"zero"`, or a path
  (`.npy` / `.json` / raw f32).

Because the table is hash-addressed, rows are shared with other n-grams.
The tool reports collisions and blends by default instead of clobbering.

## Usage

```bash
# 1. produce an overlay (tiny; does not touch the GGUF)
python3 inject.py \
  --gguf /path/to/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
  --knowledge examples/knowledge.json \
  --mode overlay --out knowledge.plepatch --report plan.json

# 2a. run llama.cpp with the overlay applied in memory (recommended)
LLAMA_PLE_OVERLAY=knowledge.plepatch ./build/bin/llama-completion \
  -m /path/to/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf -p "..." 
# or place the overlay next to the model as "<model>.gguf.plepatch"

# 2b. or rewrite the GGUF outright (needs ~54 GB free)
python3 inject.py --gguf ... --knowledge ... --mode materialize --out ./injected

# 2c. or patch the GGUF in place (writes an undo file)
python3 inject.py --gguf ... --knowledge ... --mode in-place
```

## The llama.cpp patch (COW overlay)

`llama_patch/ple-overlay.patch` adds `src/llama-ple-overlay.{h,cpp}` and calls
`llama_model_loader::apply_ple_overlay()` right after `init_mappings()`. It
maps the patched rows `MAP_PRIVATE | MAP_FIXED` over the model's read-only
`MAP_SHARED` mapping: the kernel copies only the touched pages on first write,
so the GGUF on disk is never modified and other processes sharing the mapping
are unaffected. Overlay discovery: `$LLAMA_PLE_OVERLAY`, else
`<first-shard>.plepatch`.

### Overlay file format (`.plepatch`)

```
offset  size  field
0       8     magic "PLEOVLY1"
8       2     version (1)
12      4     qtype (GGML type id, e.g. 8 = Q8_0)
16      8     row_dim (160)
24      8     bytes_per_row (170 for Q8_0)
32      8     n_rows
40      8     manifest_len
48      8     reserved
56      72    tensor name (NUL-padded)
128     var   manifest JSON (provenance)
...     8+var row records: u64 row_id + bytes_per_row payload
```

## Verification

- `tests/test_ple.py`: 8/8 pass — hash matches C++ golden vectors on 18
  windows, EOS-reset semantics, GGUF-BPE == HF tokenizer, Q8_0 roundtrip
  byte-identical, untouched rows bit-exact after patch, overlay roundtrip.
- C++ COW unit test: patched rows visible through the read-only mapping,
  unpatched rows intact, file on disk byte-identical.
- Loader E2E on the real 188 GB model: 26 rows applied, 0 mismatches, disk
  untouched; env-var and sibling-file discovery both work.
- Semantic test: the *runtime* hash (exact copy of `set_input`) on the trigger
  tokens `[760,6511,314,9338,369]` lands on exactly the 16 overlay rows, and
  those rows carry the injected vectors; stored bytes ==
  `requant(blend(orig, rand, 0.3))` byte-exact.

## Requirements

- Python 3.10+: `pip install -r requirements.txt` (`numpy`, `gguf`,
  `tokenizers` — the last one optional at runtime, GGUF BPE is the fallback).
- llama.cpp with qwen4exp support: apply `llama_patch/ple-overlay.patch` to
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)@`6c84c7d5`
  (or a descendant with qwen4exp) and build with CMake.

## Quick start

```bash
pip install -r requirements.txt
# put the GGUF shards in model/ (or pass any shard path)
python3 inject.py --gguf model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
        --knowledge examples/knowledge.json --mode overlay \
        --out my.plepatch --dry-run     # plan first
python3 tests/test_ple.py              # 8/8 must pass
```

Full walkthrough (including concept vectors and table inspection):
[`examples/README.md`](examples/README.md).

## License

MIT — see [LICENSE](LICENSE). The llama.cpp patch inherits llama.cpp's MIT
license; Qwen3.8-Flash-Next weights are subject to their own license and are
not distributed here.
