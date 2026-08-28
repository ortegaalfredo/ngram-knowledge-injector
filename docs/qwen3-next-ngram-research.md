# "Qwen3-Next + n-gram" — Research Report
Compiled: 2026-08-27 · Sources: primary (Qwen tech report PDF, HF configs, SGLang/vLLM/NVIDIA docs, arXiv)

## 0. Disambiguation first (this is the crux of the query)
There are **two unrelated "n-gram" things** in the Qwen-Next story. Most confusion comes from mixing them:

| | (A) N-gram **Embedding** (architecture) | (B) N-gram **Speculative Decoding** (inference trick) |
|---|---|---|
| What | A 51B-parameter hash-addressed learned lookup table that is part of the *trained weights* | A draft-token proposer that pattern-matches previously seen text |
| Model | **Qwen3.8-Flash-Next** (released 2026-08-26), Qwen4-arch preview | Any model; used on Qwen3-Next / Qwen3.6 in llama.cpp, vLLM, TRT-LLM |
| Age | New in Flash-Next | Existed long before; unrelated to Qwen |
| On Qwen3-Next-80B-A3B (2025)? | **NO** — verified: `config.json` `model_type: qwen3_next` has zero `ngram`/`ple` keys | Yes, community usage |

**Verified fact:** the original `Qwen/Qwen3-Next-80B-A3B-Instruct` config contains no n-gram or PLE fields at all. The n-gram embedding is new in **Qwen3.8-Flash-Next** (`model_type: qwen4_exp`).

---

## 1. What the n-gram embedding actually is (Qwen3.8-Flash-Next)

### 1.1 Verified config (`https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`)
```
ngram_size                    = 3
heads_per_ngram               = 8
ngram_vocab_size_base         = 20,000,000
ple_embed_dim                 = 2560
ple_layer_ids                 = [2]        # single layer, 2nd decoder block
ple_conv_kernel_size          = 4
split_ngram_parts             = 128        # table sharded into 128 pieces
make_ngram_vocab_size_divisible_by = 128
vocab_size                    = 248,320
```

### 1.2 Mechanism (SGLang day-0 blog, arithmetically verified)
For token `x_t`:
- **8 two-gram hash heads** on `(x_{t-1}, x_t)` + **8 three-gram hash heads** on `(x_{t-2}, x_{t-1}, x_t)`
- → **16 embedding row IDs**; each row = **160 values** → concat `E_t ∈ R^2560`
- Table shape: 16 heads × 20,000,000 rows = **320,000,000 rows × 160 = 51.2B params**
  - GGUF dump confirms exactly: `per_layer_token_embd.weight | 160, 320001536, 1, 1`
  - 51.2B × 2 bytes = **95.37 GiB in BF16** (matches SGLang's "95.4 GiB")
- Injection: gated into the 4 Gated-Residual (HC) branches before HC Mix:
  `g_t = Gate(Norm(Q_t), Norm(K_t))`, `U_t = g_t ⊙ V_t`,
  `Δ_t = U_t + SiLU(DWConv(RMSNorm(U_t)))`, `R̃_t = R_t + Δ_t`
- Request-local state: last 2 token IDs + short-conv history `[10240, 9]`

### 1.3 The key systems property
Addressing is **deterministic and data-independent** → row IDs are known before compute runs → the table can live in **host RAM (or even NVMe)** and be **asynchronously prefetched**, overlapping with layer-1 compute. Only **16 rows = 5,120 bytes/token** are touched. This is why it is a *capacity* axis, not a *compute* axis.

### 1.4 Parameter accounting
125B MoE + 51B n-gram + 4B MTP = **~180B total, 6B active**. BF16 checkpoint 335.28 GiB; FP8 172.78 GiB.

---

## 2. Qwen's own ablations (tech report §2.3) — the most interesting part
Source: `github.com/QwenLM/Qwen3.8-Flash-Next/tech_report.pdf` (2026-08-26), §2.3, Tables 7–9. 300 tokens/active-param throughout.

**Table 7 — Placement (fixed n-gram budget, avg over 10 benchmarks):**
| Placement | Avg |
|---|---|
| none | 45.44 |
| 1st | 47.30 |
| **2nd** | **47.94** |
| 3rd | 46.76 |
| 10th | 46.62 |
| 15th | 47.37 |
| 25th | 47.40 |
| 2nd+15th | 47.01 |
| 2nd+25th | 47.75 |

Findings: **a single layer is sufficient**; splitting the budget across layers gives no consistent gain. Placement is largely insensitive to attention type. Layer 2 was chosen *for systems reasons* — it lets host-memory prefetch overlap with layer-1 compute.

**Table 8 — Fixed total parameter budget (experts reduced to pay for the table):** loss is lowest at 10× vocab (25% of budget), but **downstream benchmarks show no clear improvement over the MoE-only baseline**. The loss optimum ≠ accuracy optimum.

**Table 9 — Free budget (table is extra parameters):**
| Vocab scale | Loss | MMLU | MATH | C-Eval | CMMLU |
|---|---|---|---|---|---|
| none | 1.585 | 62.78 | 32.52 | 66.91 | 68.10 |
| 20× | 1.553 | 64.14 | 37.38 | 71.75 | 72.29 |
| 50× | 1.541 | 64.71 | 37.32 | 72.12 | 72.48 |
| 100× | 1.534 | 64.70 | 35.87 | 73.75 | 72.73 |
| 200× | 1.526 | 64.85 | 35.34 | 74.94 | 73.24 |

**The honest headline: loss falls monotonically, but downstream accuracy does not follow.** MATH *peaks at 20× then degrades* (37.38 → 35.34). Chinese benchmarks (C-Eval/CMMLU) are the only ones that improve monotonically. Qwen explicitly flags this loss/accuracy disagreement in the intro.

Also: Qwen tried token normalization for vocab compression, non-uniform allocation across n-gram orders, and frequency-based slot partitioning — **"we observed no consistent performance gains."**

Optimizer note: the n-gram table trains on **Adam with weight decay disabled** (Muon handles 2-D linear maps; AdamW handles embeddings/output head/router/GR low-rank).

Stability: full Flash-Next recipe (GR + n-gram) lowered loss by 0.058 over a Qwen3.5+Muon baseline at 276B tokens.

---

## 3. Lineage — who actually invented this
Qwen's §2.3 citation list (verified in the PDF reference list):
- **DeepSeek "Engram"** — Cheng et al., *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs*, **arXiv:2601.07372** (12 Jan 2026, v2 12 Jul 2026; ACL 2026). O(1) hashed n-gram lookup, multi-head hashing, tokenizer compression, context-aware gating, depthwise causal conv (kernel 4, dilation = max n-gram order), multi-branch integration. U-shaped sparsity-allocation law; scaled to 27B. Gains: MMLU +3.4, CMMLU +4.0, BBH +5.0, HumanEval +3.0, MATH +2.4, MQ-NIAH 84.2→97.0. Engram's slots scale as a **strict power law** (Fig. 3).
- **Gemma 3n / Gemma 4 "Per-Layer Embeddings" (PLE)** — Google DeepMind. This is where SGLang's "PLE" name comes from. Gemma's PLE is keyed on token ID and feeds *every* layer; Qwen's is n-gram-keyed and in **one** layer.
- **SCONE** (Yu et al. 2025) — Scalable, Contextualized, Offloaded, N-gram Embeddings.
- **OverEncoding** (Huang et al. 2025), **N-Grammer** (Roy et al. 2022), **X-Gram** (Chen et al. 2026, arXiv:2604.21724), **STEM** (Sadhukhan et al. 2026, arXiv:2601.10639), *Scaling Embeddings Outperforms Scaling Experts* (Liu et al. 2026, arXiv:2601.21204), **RWKV-V8 DeepEmbed**.
- Community notes **Longcat** as an earlier public user of the idea.

So: **Qwen did not invent it — it is the first large open-weight production-scale deployment of it** (51B table vs Engram's 27B).

Follow-up critique already in literature: **Tensorized Engram / TN-gram** (arXiv:2606.08347, Jun 2026) — argues per-order hash tables cause **hash collisions** and prevent nested n-grams sharing latents; CP-factorization matches/beats Engram with far fewer params. arXiv:2601.21204 likewise: "hash collisions force a single embedding vector to superimpose the semantics of multiple distinct n-grams."

---

## 4. Serving / deployment reality
- **vLLM** (`vllm/vllm-openai:qwen38-flash-next`): `VLLM_PLE_CPU_OFFLOAD=1` keeps the table in host RAM with async row prefetch. Needs ≥51 GB host RAM + headroom. Offload is NVIDIA-only. TP8 plain is incompatible with FP8 (128-wide quant blocks) → use TEP8.
- **SGLang** (day-0): table shard → **pinned host memory**, Triton UVA kernel gathers the 16 rows/token into a small BF16 GPU buffer, dedicated CUDA stream overlaps the gather with decoder block 1. **Measured on H200 TP4 + MTP:** target weights 83.91 → **60.45 GiB/GPU (−23.46 GiB)**, KV capacity 1.84M → **3.28M tokens (+78.5%)**, throughput change **−0.07% geomean**, outputs bit-exact. Only the 1-layer MTP *draft* disables PLE.
- **llama.cpp**: PR **#27742** (`qwen4exp`, danielhanchen/Unsloth) — still **open/unmerged** as of 2026-08-27; 23 files / ~2,874 lines, no new ggml ops. PLE = `per_layer_token_embd.weight`, host-side row indices in `set_input` then `ggml_get_rows`. Correctness: wikitext-2 PPL **4.0068 vs 4.0126** reference, 98.0% top-1 agreement.
  - **Metal gotcha:** any shard containing a GPU tensor gets fully wired. If the n-gram table shares shards with weights it becomes non-swappable → OOM. Fix = repack the table into its own shard (M64 builds do this). Measured on M5 Max 128 GB: wired 114.4 GiB → **90.8 GiB** after repacking.
  - Quants keep the table high-precision: e.g. 3.84bpw build = MoE body at **IQ1_M (1.75bpw)** but table at **Q5_1 (38.4 GB, 28.9% of params)**. Unsloth: "1-bit is 75GB and uses 4-bit for the Ngram/PLE."
- **Real-world local numbers:** M1 Max 64 GB, table on SSD → **17.6 tok/s decode, 181.7 tok/s pp512** — i.e. a 125B model running at the same speed as a 27B dense model on the same box. DGX Spark: NVMe-paged table, ~80 tok/s prefill / 12 tok/s decode, ~80–90 GiB resident.
- **Datacenter:** GB300 NVL72 >16K tok/s/GPU, >200 tok/s/user.

---

## 5. Thread (B): n-gram *speculative decoding* — separate, and results are mixed
- **TensorRT-LLM tech blog** (Llama-4-Scout-17B, 8×B200 FP8): accepted length ~**1.37** on first-turn chat → **10–60% E2E speedup**; second turn AL **1.66** → **30–90%**; translation AL **>4.0**, up to **70%** latency cut. `spec_decode_algo=AUTO` heuristic: <15% iteration overhead for ~15% E2E gain. Works best at **low batch/low concurrency**.
- **vLLM**: `{"method":"ngram"}` prompt/prefix matching.
- **llama.cpp**: `--spec-type ngram-cache` / `ngram-mod` / `ngram-simple`.
  - Positive claims: Qwen3.6-27B 22 → 56 tok/s stacking MTP + ngram-mod; viral ~10x / 136 tok/s claims on Qwen3.6-27B — **these are repetition-heavy iterative-coding sessions, not open-ended chat.**
  - **Negative, rigorous:** thc1006's 19-config matrix, Qwen3.6-35B-A3B + RTX 3090 (post PR #19493): **no** spec-decode mode (ngram-cache, ngram-mod, or vocab-matched Qwen3.5-0.8B draft) achieves net speedup; mean decode **3–12% slower**.
  - llama.cpp issue **#23184**: `draft-mtp,ngram-mod` run *independently*, don't share context, no benefit over MTP alone; CUDA OOM reported (#23154). Closed as not planned.
- **For Qwen3.8-Flash-Next specifically, n-gram spec-decoding is largely moot**: the model ships a built-in **4B MTP head** (`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`), which is a strictly better drafter.

---

## 6. Assessment
**Why it matters:** it is a genuine third axis. MoE = conditional *compute*; n-gram embedding = conditional *memory*. Because lookups are deterministic, capacity can be pushed into cheap, slow, huge host DRAM instead of scarce HBM — a real answer to the memory wall, and it explains why a 180B model is now practical on 64–128 GB Macs/Sparks.

**Caveats worth carrying:**
1. Qwen's own data shows **loss and accuracy diverge**; MATH regresses past 20× vocab. The win is strongest on knowledge/Chinese benchmarks, weakest on math.
2. Under a **fixed parameter budget it did not beat MoE-only** — the table only pays off as *extra* parameters.
3. **Hash collisions** are a known open weakness (TN-gram, arXiv:2601.21204); Qwen tried several collision mitigations and reported no consistent gains.
4. The table is **static/read-only** — no knowledge update without retraining.
5. It adds a **systems dependency**: the benefit assumes DRAM/NVMe offload works well. Poorly integrated (e.g. Metal wiring) it becomes a liability.
6. Qwen calls it an **"experimental preview"**; the production Qwen3.8-Flash API ($0.16/$0.47 per M tokens, 1M ctx) is built on it, but Qwen4 itself is still to come.
