# llama.cpp PLE overlay patch

Patch for [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
(commit `6c84c7d5d8833c6e0df69628f75a0f599797934e`, branch with `qwen4exp`
support — the fork used here was named `llama.cpp-NLTM`) that loads a
`.plepatch` overlay **in memory** at model-load time.

## What it does

- Adds `src/llama-ple-overlay.{h,cpp}`:
  - `llama_ple_overlay_read_index(path)` — parses the 128-byte-header
    `.plepatch` format (row id → raw Q8_0 bytes; format documented in the
    top-level [README](../README.md)).
  - `llama_ple_overlay_for_model(loader)` — discovery: `$LLAMA_PLE_OVERLAY`
    env var, else `<first-shard>.plepatch` sibling.
  - `llama_ple_overlay_apply(mapping, ov, path, data_off, n_rows)` —
    `mmap(..., MAP_PRIVATE | MAP_FIXED)` the payload pages over the model's
    read-only mapping. The kernel copies pages on write (COW), so:
    - the GGUF on disk is **never** modified,
    - other processes sharing the same file mapping are unaffected,
    - only the touched 4 kB pages cost extra RAM.
- Calls `apply_ple_overlay()` from `llama_model_loader::init_mappings()`
  (in `src/llama-model.cpp` right after mappings exist) and tracks the extra
  file in the loader's `file_paths` bookkeeping (`src/llama-model-loader.cpp`).

## Apply

```bash
cd <your llama.cpp checkout>
git apply --check /path/to/ngram_tool/llama_patch/ple-overlay.patch
git apply        /path/to/ngram_tool/llama_patch/ple-overlay.patch
cmake -B build -DGGML_NATIVE=ON && cmake --build build -j
```

> Rebased cleanly on `6c84c7d5` for this repo; earlier/later revisions may
> need small context fixes.

## Tests

```bash
bash llama_patch/build_tests.sh /path/to/llama.cpp-fork   # -> /tmp/test_*
/tmp/test_cow_overlay                                     # COW unit test (fake file)
LLAMA_PLE_OVERLAY=k.plepatch /tmp/test_loader_overlay model.gguf k.plepatch env
/tmp/test_semantic model.gguf k.plepatch 760,6511,314,9338,369
```

| Test | Proves |
|---|---|
| `test_cow_overlay` | patched rows visible through a read-only mapping, unpatched rows intact, file on disk byte-identical |
| `test_loader_overlay` | real loader applies the overlay on the real model in 3 discovery modes; verifies disk bytes differ |
| `test_semantic` | the runtime hash (`set_input` copy) lands on exactly the overlay rows, which carry the injected vectors |

Memory note: `init_mappings` may pre-fault mappings (`MAP_POPULATE`) when
called with eager mode — on a 188 GB model this OOMs smaller machines. The
test binaries here map lazily (peak RSS ≈ 41 MB).
