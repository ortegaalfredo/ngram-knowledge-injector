# PLE Knowledge Injection — Step-by-Step Guide

How to teach a Qwen3.8-Flash-Next (`qwen4exp`) GGUF new behavior by writing into
its PLE n-gram embedding table — without re-quantizing and without touching the
GGUF on disk (unless you want a materialized copy).

- **Part 1** — writing / updating `knowledge.json`
- **Part 2** — loading the knowledge into llama.cpp (overlay sidecar)
- **Part 3** — generating a fully patched GGUF model (materialize mode)
- **Part 4** — deriving a real concept-direction vector (the exact recipe)
- **Part 5** — validating everything + troubleshooting

> Looking for the smallest complete demo? [`MyColor/`](MyColor/README.md)
> injects a chosen color into the prompt `My color is` with a single-entry
> knowledge file, a one-line overlay build, and a one-line curl probe.

---

## 0. How the mechanism works (30 seconds)

The model has one shared table `per_layer_token_embd.weight`
(`160 x 320001536`, Q8_0). At inference, for every token the runtime hashes the
local context (current token + up to `ngram_size-1` predecessors, here
`ngram_size=3`) into `n_heads=16` rows of that table and **adds the gathered
vectors into the residual stream** at the PLE layer(s) (`qwen4exp.ple.layers`).

`inject.py` computes exactly those rows for the trigger contexts you list and
writes your vectors into them. Consequences:

- influence fires **exactly when the trigger context appears**;
- rows are **hash-shared** — every n-gram that collides on a row gets the same
  vector, so the tool *blends* by default instead of clobbering;
- the trigger must be tokenized **exactly like inference** — use raw
  completions that match your intended prompt byte-for-byte, *not* chat
  templates (`<|im_start|>` wrappers change the tokens and therefore the rows).

---

## 1. Updating `examples/knowledge.json`

### Step 1.1 — understand the schema

```jsonc
{
  // applied to every entry unless overridden
  "defaults": { "at": "last", "heads": "all", "orders": "all",
                "op": "blend", "alpha": 0.5 },

  "entries": [
    {
      "trigger": "Q: What is your favorite color? A:",  // REQUIRED: context to arm
      // one of:
      "vector": "random",              // seeded gaussian (needs "seed")
      // "vector": [0.1, ...],         // literal 160 floats
      // "vector": "red.vec.npy",      // file: .npy | .json | raw f32 (see Part 4)
      "op": "blend",                   // blend|set|add|scale|copy_from|zero
      "alpha": 0.8,                    // blend mix (0..1); higher = stronger
      "scale": 1.0,                    // multiplier for "add"
      "seed": 42,                      // for "vector": "random"
      "at": "last",                    // last|first|all|<token index>
      "heads": [0, 8],                 // head subset or "all" (16 total:
                                        //   heads 0-7 = bigram, 8-15 = trigram)
      "orders": [3],                   // 2 and/or 3 -> restrict head order
      "prefix": "",                    // prepended to trigger for hashing only
      "copy_from": "...",              // for op "copy_from": source trigger
      "note": "why this entry exists"
    }
  ]
}
```

Op semantics (applied per row, `existing` = current row content, possibly
already modified by an earlier colliding entry):

| op          | result                                            |
|-------------|---------------------------------------------------|
| `blend`     | `existing*(1-alpha) + target*alpha`               |
| `set`       | `target` (overwrites; use with care)              |
| `add`       | `existing + target*scale` (steering-style push)   |
| `scale`     | `existing * scale`                                |
| `copy_from` | target = mean of the rows the runtime would hash for `copy_from`'s last token — "make this context feel like that context" |
| `zero`      | all zeros (delete the signal the table carries)   |

### Step 1.2 — add your entry

Example (already in [`knowledge.json`](knowledge.json)) — make the model's
answer to the color question biased by an injected vector:

```json
{
  "trigger": "Q: What is your favorite color? A:",
  "vector": "random", "seed": 42, "op": "blend", "alpha": 0.8,
  "note": "fires on the color question; biases the color the model names"
}
```

Rules of thumb:

- **trigger** = the full raw completion up to (and including) the point where
  the next token is the one you want to influence.
- **`at: "last"`** (default) is almost always what you want: the vector lands
  at the position the next token is generated from.
- **strength**: `blend` `alpha` 0.3 = subtle nudge, 0.8 = strong push. For
  `add`, `scale` 0.5–2.0 is a sane range.
- **narrow the blast radius** on long triggers: `"orders": [3]` touches only
  trigram heads (8 rows), `"heads": [0, 8]` touches one bigram + one trigram
  head (2 rows). Fewer rows = fewer collisions = cleaner experiment.

### Step 1.3 — always dry-run first

```bash
python3 inject.py --gguf model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
        --knowledge examples/knowledge.json --mode overlay --dry-run
```

Read the plan:

```
plan: 34 unique rows to write (34 touched by entries)
note: 8 rows touched by >1 entry (hash collisions merge them)
  - {"trigger": "Q: What is your favorite color? A:", "op": "blend",
     "n_tokens": 10, "positions": [9], "heads": 16, "orders": "all",
     "target_rows": 16}
```

Check: `n_tokens` looks like the tokenization you expect, `positions` is where
the vector fires, `target_rows` is the row count. Nothing is written in
dry-run. `--report plan.json` dumps the same info as JSON.

---

## 2. Loading the knowledge into llama.cpp

Build the sidecar overlay (tiny file: row id + raw Q8_0 bytes per row):

```bash
python3 inject.py --gguf model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
        --knowledge examples/knowledge.json --mode overlay \
        --out build/knowledge.plepatch --report build/plan.json
```

The fork applies it at load time via `apply_ple_overlay()` — the patched rows
are mapped `MAP_PRIVATE` **in memory over the read-only GGUF mapping**; the
files on disk are never modified. Three ways to point the loader at it:

NOTE: The overlay has these limitations:
       1. Quantization must be Q8_0, as other quantizations of the PLE tamble are not currently implemented
          The model used for the demonstration is: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/Q8_0
       2. File must be MMAPED into memory, as the non-mmap modification is not implemented yet.
      This mean you require about 150GB of VRAM and at least 64GB of RAM to run the demonstration,


### 2a. Environment variable (recommended)

```bash
export LD_LIBRARY_PATH=$PWD/<llama.cpp-fork>/build/bin
LLAMA_PLE_OVERLAY=$PWD/build/knowledge.plepatch \
  <llama.cpp-fork>/build/bin/llama-completion \
  -m model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
  -p "Q: What is your favorite color? A:" -n 16
```

Look for the confirmation line: `apply_ple_overlay: applied 34 PLE overlay row(s)`.

### 2b. Sibling file

`cp build/knowledge.plepatch model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf.plepatch`
— discovered automatically (delete it afterwards to go back to stock behavior).

### 2c. Explicit path in your own code

Call `llama_model_loader::apply_ple_overlay()` after `init_mappings()` with
`ml.ple_overlay = llama_ple_overlay_read_index(path)` — see
[`llama_patch/test_loader_overlay.cpp`](../llama_patch/test_loader_overlay.cpp).

> **Memory**: mapping is lazy, so loading costs ~nothing extra. *Generation*
> touches the 54 GB table — run inference only on a machine with RAM to spare
> (rule of thumb: free RAM > model size + 60 GB), and never pre-fault the
> mappings (`init_mappings(true)` sets `MAP_POPULATE` and will OOM a 125 GB
> box; the test binaries in `llama_patch/` are already fixed to lazy mapping).

---

## 3. Generating a patched GGUF (materialize mode)

When you want a portable model file (no sidecar, any llama.cpp):

```bash
python3 inject.py --gguf model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
        --knowledge examples/knowledge.json --mode materialize \
        --out model/MyColorModel.gguf
```

What it does, step by step:

1. hardlinks every shard into the output set (near-zero disk, seconds);
2. makes one real copy of the shard that carries the PLE table (shard 3 here,
   ~51 GB — needs that much free disk);
3. rewrites only the planned rows (`row * bytes_per_row` seeks) in that copy.

Result: `model/MyColorModel-0000{1..6}-of-00006.gguf`, byte-identical to the
original except the injected rows. Run it like any model. (In-place mode
`--mode in-place` also exists; it patches the original GGUF directly and writes
an undo file — prefer materialize.)

---

## 4. Deriving a concept-direction vector — the exact recipe

Note: This is a work in progress and the concept vectors derived from these tools
basically don't work yet. The current way to derive a correct concept vector, that
is, a color, or a name that gets correctly inferenced after the trigger, is via brute-force: 
Just test random vectors until one works. 

Now these are two experimental ways to deterministically capture a concept vector:

`"vector": "random"` proves the plumbing but is not a concept. To make the
model *prefer red*, the injected 160-d vector should point toward "red" in the
per-layer-input space the table lives in. Two methods, best first.

### Method A — capture the model's own vector for a concept context (tool-native, no GPU)

The table row for a context *is* the model's learned per-layer signal for that
context. `op: "copy_from"` already uses this ("make this context feel like
that one"). To export it as a reusable `.npy`, [`capture_vector.py`](capture_vector.py)
does exactly what the runtime does — hash the context, read the rows, average
the heads:

```bash
# 1. capture the direction for the concept (use several rich contexts)
python3 examples/capture_vector.py \
    --gguf model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
    --text "My favorite color is red. Red is the color of" \
    --out examples/red.vec.npy
```

The script (see file) is 30 lines of `ple_core`: `encode -> rows_for_sequence
-> read_rows(last position) -> mean over heads -> save f32 `.npy`. Repeat for
more concept sentences and average the `.npy` files, or pass
`--mean a.npy b.npy` if you captured several.

```bash
# 2. reference it from knowledge.json — "vector" paths resolve relative to
#    the knowledge file's own directory (absolute paths work too)
{
  "trigger": "Q: What is your favorite color? A:",
  "vector": "red.vec.npy", "op": "add", "scale": 1.0,
  "note": "push toward red, direction captured from the model itself"
}
```

```bash
# 3. dry-run, rebuild the overlay, ask (Part 2), tune --scale / alpha
```

Because the vector came from the table itself, it is *exactly* in-distribution:
same 160-d space, same Q8_0-compatible magnitude range. This is the supported
path in this repo.

### Method B — difference-in-means activation steering (proper "concept direction", needs a runnable model)

The textbook way to get a concept direction:

1. **Hook the injection point.** Record the residual vector that the PLE add
   receives at layer `ple.layers[0]` (= 1) for the **last token** of each
   prompt. In the fork this is the buffer `set_input()` fills
   (`llm_graph_input_ple`, see `llama_patch/ple-overlay.patch`); in any HF
   implementation it is the layer-1 per-layer input.
2. **Collect two prompt sets** (>=32 each, diverse):
   - *with concept*: `My favorite color is red.`, `The red apple is…`,
     `Red is the color of…`, … (end every prompt at the same syntactic
     position)
   - *neutral*: same skeletons with the concept word removed/swapped for
     unrelated fillers.
3. **Reduce**: `dir = mean(activations_with) - mean(activations_neutral)`,
   then L2-normalize. (Optionally whiten with the pooled covariance — the
   projection form `Σ⁻¹(μ_with − μ_neutral)` is more robust.)
4. **Project into 160-d** if your hook gave a higher-dim residual: fit a linear
   map `W` from token embeddings to per-layer inputs on ~1k tokens (least
   squares) and take `W @ dir` — or simply reuse Method A vectors as the
   basis. Save as float32 `.npy`, length must equal 160 exactly (validated at
   plan time).
5. **Apply with `add`**, not `blend`: `"vector": "red-dir.vec.npy",
   "op": "add", "scale": 1.5`. Normalized direction × scale = steering
   strength; start at ~1.0 and sweep.

Pseudocode for the hook (PyTorch, any model exposing per-layer inputs):

```python
acts_with, acts_neutral = [], []
def hook(module, inp, out):
    acts.append(inp[0][:, -1, :].detach().cpu())   # last token
for prompts, store in [(WITH, acts_with), (NEUTRAL, acts_neutral)]:
    for p in prompts: acts = store; model(p)
import numpy as np
dir = (np.mean(acts_with, 0) - np.mean(acts_neutral, 0)).ravel()
np.save("red-dir.vec.npy", (dir / np.linalg.norm(dir)).astype(np.float32))
```

Validate the direction the same way as Method A: generate with and without the
overlay and compare.

---

## 5. Validation & troubleshooting

**Always validate in this order:**

```bash
# 1. python suite (8/8) — includes C++ golden hash match on 18 windows
python3 tests/test_ple.py

# 2. build & run the C++ tests against the fork
bash llama_patch/build_tests.sh "$PWD/<llama.cpp-fork>"
export LD_LIBRARY_PATH=$PWD/<llama.cpp-fork>/build/bin
/tmp/test_cow_overlay                                            # COW unit test
LLAMA_PLE_OVERLAY=$PWD/build/knowledge.plepatch \
  /tmp/test_loader_overlay model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
  build/knowledge.plepatch env                                    # real-loader E2E
```

Expected decisive lines: `checked 34 patched rows through the mapping,
0 mismatches`, `disk bytes at patched row … differ from overlay payload: yes`,
`LOADER OVERLAY E2E PASSED`.

| Symptom | Cause / fix |
|---|---|
| No effect at generation | trigger not matched verbatim — chat template changed the tokens. Use raw completion (`llama-completion -p "…"`) with the exact trigger text. |
| `vector length N != row_dim 160` | `.npy` not 160-d; re-derive or pad via Method A capture. |
| `rows touched by >1 entry` | hash collisions — intended; entries merge in order. Split with `orders`/`heads` if unwanted. |
| `applied 0 overlay row(s)` / tensor not found | overlay built for a different model; check `table_tensor` + `table_sha1_before` in the sidecar manifest. |
| OOM during generation | table is huge; see memory note in Part 2. |
| GGUF integrity worry | rows on disk are never written by overlay mode; verify with the loader test's `disk bytes … differ: yes`. |

---

## 6. Inspecting the table — `ple_dump.py` ([`../ple_dump.py`](../ple_dump.py))

A read-only tool to see exactly which concepts are in the PLE. First, the
honest limits: the PLE table lives in a 160-d per-layer-input space **with no
LM head tied to it**, so there is no exact vector→text decoder. The tool gives
you two sound views instead:

1. **EXACT collisions** — the runtime hashes a context into specific rows; any
   probe context hashing to the same row *shares that row at inference*.
   Ground truth, not a heuristic.
2. **Cosine labeling** — rows carry learned content; comparing against labeled
   probe contexts surfaces what a row behaves like. Correlative, and every hit
   is printed with its cosine so you can judge strength.

All commands below use `--gguf model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf`
(abbreviated `<shard>`). Reads are sparse (fseek per row) — nothing loads the
54 GB table.

### Step 6.1 — what does a context read? (`context`)

```bash
python3 ple_dump.py --gguf <shard> context --text "The capital of France is"
```
```
tokens (5): [760, 6511, 314, 9338, 369]
pieces: ["'The'", "' capital'", "' of'", "' France'", "' is'"]
n_heads=16 (heads 0-7=bigram, 8-15=trigram)
position 4:
  head  0 -> row 11681901   norm=0.0979 ...
```
Shows the exact rows the model will gather for this context, per head/order.
`--at all` dumps every position; `--out` saves the vectors as `.npy`.

### Step 6.2 — build a probe set (`probes`)

```bash
python3 ple_dump.py --gguf <shard> probes --words 200 --pairs 800 --out build/probes.txt
```
Auto-generates probe contexts from the tokenizer vocabulary (single words +
random bigrams). Or write your own file — one context per line, e.g.
`red`, `Paris is the capital of`, `def main():` — domain probes beat random
probes for finding *specific* concepts.

### Step 6.3 — label a row or context (`decode`)

```bash
python3 ple_dump.py --gguf <shard> decode --row 11681901 \
    --probes build/probes.txt --probe-cache build/probe_cache.npz -k 6
# or label the rows of a context:
python3 ple_dump.py --gguf <shard> decode --text "The capital of France is" \
    --probes build/probes.txt --probe-cache build/probe_cache.npz
```
```
row 11681901 norm=0.0979 ...
  EXACT: no probe hashes here (row unprobed)
  nearest probe rows (cosine, * below 0.25):
   0.301 'our'(h5) |  0.281 'est'(h11) | ...
```
The `EXACT:` line is ground truth (which other contexts share this row — they
*do* interfere at inference). The cosine list is the correlative guess.
`--probe-cache` stores probe vectors as `.npz` so re-runs skip re-reading the
table.

### Step 6.4 — survey the whole table (`scan`, `stats`)

```bash
python3 ple_dump.py --gguf <shard> scan  --probes build/probes.txt \
    --probe-cache build/probe_cache.npz --samples 200 --seed 0
python3 ple_dump.py --gguf <shard> stats --samples 1500
```
```
norm   : min=0.058 p25=0.091 med=0.096 p75=0.102 max=0.135
dead rows (norm < 1e-4): 0 (0.00%)
```
`scan` labels random rows and reports how many got confident labels — expect a
minority with small probe sets (real output: `40/40 labeled above cos 0.25`
at a low bar, but `0/60 above 0.4` — treat margins as confidence). `stats`
shows norms/health of the table.

### Step 6.5 — raw dumps (`rows`)

```bash
python3 ple_dump.py --gguf <shard> rows --range 0:100 --out build/head.npy
python3 ple_dump.py --gguf <shard> rows --rows 11681901 --raw
```
`--range START:COUNT` dumps contiguous rows to `.npy` (row ids saved alongside
as `.ids.npy`); `--raw` prints full 160-d vectors to stdout.

### Honest interpretation guide

- `EXACT:` collisions are real sharing — two contexts sharing a row interfere
  at inference (this is what `inject.py`'s collision report manages).
- Cosines ≈0.25–0.40 with small margins are **weak evidence**; the tool tags
  them (`*`) so you don't over-read them. For sharper labels, grow the probe
  set with domain text you care about.
- A row's norm (~0.096 median here) is the signal magnitude added to the
  residual stream whenever a context hashes to it.
- To see what *your injections* did: `decode --row <row>` on the rows printed
  by `inject.py --report`, and compare against the overlay's `first5` dumps in
  the semantic test.
