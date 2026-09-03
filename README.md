# PLE n-gram knowledge injector for Qwen3.8-Flash-Next

Experimental implementation of knowledge injection into the **PLE n-gram embedding table** of a
`qwen4exp` (Qwen3.8-Flash-Next) GGUF, without rewriting the 54 GB table.

![Real-time knowledge injection example](https://raw.githubusercontent.com/ortegaalfredo/ngram-knowledge-injector/refs/heads/main/ngram-patch-example.gif)

## What this project does

Qwen3.8-Flash-Next ships with a huge built-in **fact lookup table** — an
n-gram embedding table the model uses like a long-term memory. Whenever the
words you type match something stored in that table, the matching entry is
looked up and fed straight into the model, right alongside what it "knows"
from its weights.

This project lets you **write into that memory directly**. Give it a phrase
("trigger") and a vector, and the next time the model sees that phrase it
behaves as if it had memorized whatever you injected, with no retraining or
no rewriting the 54 GB table. Because the table is addressed by hashing, we
can compute, for any phrase, exactly which rows will be read at runtime, and
patch only those.

What you produce is a small **patch file** (`.plepatch`). A patched build of
llama.cpp — [llama.cpp-NLTM](https://github.com/ortegaalfredo/llama.cpp-NLTM) —
delivers **hot-swappable knowledge injection**: it loads that patch
automatically in realtime, so you just start the server or CLI as usual and the
injected memory is live from the first token. And because the running
`llama-server` re-checks the patch file before every request, you can edit,
swap or delete it at any time — the next query runs with the new knowledge, no
model reload and no restart.

### How is this different from training?

| | This (editing the n-gram memory) | Training / fine-tuning |
|---|---|---|
| What changes | A handful of rows in the lookup table | The model's weights, via gradient descent |
| Time & cost | Seconds, on CPU, on top of the existing GGUF | Hours to days, on GPUs, plus an optimizer |
| Scope | Very narrow: the exact n-grams you target | Broad: shifts behavior across many tasks |
| Side effects | Only other n-grams that hash to the same rows | Forgetting / regression on unrelated tasks |
| Reversible | Yes — the model file on disk is never touched (overlay mode) | Effectively permanent; you keep re-training |
| Best for | Pinning specific facts or behaviors ("when I say X, answer Y") | Teaching new skills or changing general capabilities |

Caveat: currently there is not a clear way to deterministically derive the knowledge vectors that you must write into the PLE table rows.
All working examples in this project rely on finding these vectors via exhaustive search, but depending on the desired outcome, this can be efficient needing around 100-200 inference steps.

The mechanics (hash function, row addressing, GGUF metadata) are covered in
[`docs/qwen3-next-ngram-research.md`](docs/qwen3-next-ngram-research.md).

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
| `examples/MyColor/` | minimal end-to-end example: pick the model's answer to `My color is` by changing a seed (see `examples/MyColor/README.md`) |
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
# llama.cpp-NLTM auto-loads it: https://github.com/ortegaalfredo/llama.cpp-NLTM

# 2b. or rewrite the GGUF outright (needs ~54 GB free)
python3 inject.py --gguf ... --knowledge ... --mode materialize --out ./injected

# 2c. or patch the GGUF in place (writes an undo file)
python3 inject.py --gguf ... --knowledge ... --mode in-place
```

## The llama.cpp patch (COW overlay)

The patch produced by `inject.py` is loaded by
[**llama.cpp-NLTM**](https://github.com/ortegaalfredo/llama.cpp-NLTM), a build
of llama.cpp that applies it automatically in realtime: at model load it maps
the patched rows over the table in memory, so the injected knowledge is active
for every request with no extra flags and no changes to the model file.

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

## Limitations

- Only **Q8_0** has been tested. Other quantizations might work but are not
  implemented/verified yet.
- The model must be **mmap'd into RAM** (`--no-mmap` is not supported — the
  overlay patches the memory-mapped table pages).

## Requirements

- Python 3.10+: `pip install -r requirements.txt` (`numpy`, `gguf`,
  `tokenizers` — the last one optional at runtime, GGUF BPE is the fallback).
- llama.cpp with qwen4exp support and the overlay patch applied — easiest is
  [llama.cpp-NLTM](https://github.com/ortegaalfredo/llama.cpp-NLTM), which
  already ships it and auto-loads the `.plepatch` at model load. To build it
  yourself instead, apply `llama_patch/ple-overlay.patch` to
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)@`6c84c7d5`
  (or a descendant with qwen4exp) and build with CMake.

## Quick start

```bash
pip install -r requirements.txt

# 1. run the test suite — works WITHOUT any model: if no GGUF is found it
#    auto-builds a synthetic stand-in (real hash constants, sparse 54 GB
#    table that occupies ~3 MB) and validates against the committed C++
#    golden vectors
python3 tests/test_ple.py              # 8/8 must pass

# 2. inject — put the GGUF shards in model/ (or pass any shard path)
python3 inject.py --gguf model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
        --knowledge examples/knowledge.json --mode overlay \
        --out my.plepatch --dry-run     # plan first
```

Full walkthrough (including concept vectors and table inspection):
[`examples/README.md`](examples/README.md).

## License

MIT — see [LICENSE](LICENSE). The llama.cpp patch inherits llama.cpp's MIT
license; Qwen3.8-Flash-Next weights are subject to their own license and are
not distributed here.
