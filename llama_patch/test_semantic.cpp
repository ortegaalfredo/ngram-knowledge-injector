// Semantic end-to-end: run the runtime PLE hash (exact copy of
// llm_graph_input_ple::set_input) on a trigger, then read the hashed rows
// through the COW overlay mapping and confirm they carry the injected vector.
#include "llama-model-loader.h"
#include "llama-ple-overlay.h"
#include "llama.h"
#include "ggml.h"
#include "llama-hparams.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>

static const uint64_t M64 = 0xFFFFFFFFFFFFFFFFull;

// Q8_0 block: fp16 scale + 32 int8
static void deq_q8_0_row(const uint8_t * p, int n, std::vector<float> & out) {
    out.clear();
    for (int b = 0; b < n; b += 32) {
        uint16_t su; memcpy(&su, p, 2); p += 2;
        ggml_fp16_t h; memcpy(&h, &su, 2);
        float s = ggml_fp16_to_fp32(h);
        for (int j = 0; j < 32; j++) out.push_back(s * (float)((int8_t)*p++));
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <overlay> <id0,id1,...>\n", argv[0]); return 2; }
    std::string model = argv[1], ov_path = argv[2];
    std::vector<int64_t> ids;
    { const char * s = argv[3]; while (*s) { ids.push_back(strtoll(s, (char**)&s, 10)); if (*s==',') s++; } }

    llama_ple_overlay ov = llama_ple_overlay_read_index(ov_path.c_str());

    std::vector<std::string> splits;
    llama_model_loader ml(nullptr, nullptr, nullptr, model, splits, nullptr,
                          LLAMA_LOAD_MODE_MMAP, false, true, false, nullptr, nullptr);

    // read the PLE constants straight from the GGUF metadata (like load_hparams)
    uint32_t ngram=0, per_gram=0, eos=0;
    std::array<uint64_t, LLAMA_MAX_PLE_NGRAM> mult{};
    std::array<uint64_t, LLAMA_MAX_PLE_HEADS> offs{}, vsz{};
    ml.get_key(LLM_KV_PLE_NGRAM_SIZE, ngram);
    ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, per_gram);
    ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID, eos);
    ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, mult);
    ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS, offs);
    ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, vsz);
    const int64_t n_heads = (int64_t)(ngram - 1) * per_gram;
    printf("runtime ple: ngram=%u per_gram=%u n_heads=%lld eos=%u\n", ngram, per_gram, (long long)n_heads, eos);

    // mappings + overlay
    // lazy mapping (no MAP_POPULATE/WILLNEED): this test touches only the
    // hashed rows; prefetching would fault in all ~179 GB of shards and OOM
    ml.init_mappings(false, nullptr);
    ml.ple_overlay = ov;
    ml.apply_ple_overlay();

    const auto * w = ml.get_weight(ov.tensor_name.c_str());
    uint8_t * base = (uint8_t*) ml.mappings.at(w->idx)->addr() + w->offs;
    const size_t bpr = ov.bytes_per_row;

    // ---- runtime hash for the LAST token of the trigger (exact set_input logic) ----
    const int64_t n = (int64_t) ids.size();
    const int64_t p = n - 1;
    const int64_t n_prev = ngram - 1;
    // ctx[0]=tok[p], ctx[s]=tok[p-s] (EOS if before start)
    std::vector<int64_t> ctx(ngram);
    ctx[0] = ids[p];
    for (int64_t s = 1; s < ngram; s++)
        ctx[s] = (p - s >= 0) ? ids[p - s] : (int64_t) eos;

    std::vector<int64_t> rows(n_heads);
    for (int64_t g = 2; g <= (int64_t)ngram; g++) {
        uint64_t mixed = (uint64_t) ctx[0] * mult[0];
        for (int64_t j = 1; j < g; j++) mixed ^= (uint64_t) ctx[j] * mult[j];
        const int64_t hbase = (g - 2) * per_gram;
        for (int64_t gg = 0; gg < (int64_t)per_gram; gg++) {
            const int64_t h = hbase + gg;
            rows[h] = (int64_t)(mixed % vsz[h] + offs[h]);
        }
    }
    printf("runtime rows for last token: ");
    for (auto r : rows) printf("%lld ", (long long) r);
    printf("\n");

    // every hashed row must be present in the overlay
    size_t in_ov = 0;
    for (auto r : rows) for (const auto & orow : ov.rows) if ((int64_t) orow.row == r) { in_ov++; break; }
    printf("hashed rows present in overlay: %zu / %lld\n", in_ov, (long long) n_heads);

    // read each hashed row through the COW mapping, dequantize, print norm
    // and confirm it differs from the on-disk (original) bytes
    FILE * disk = fopen(ml.file_paths.at(w->idx).c_str(), "rb");
    int changed = 0;
    for (size_t i = 0; i < rows.size() && i < 3; i++) {
        const uint8_t * row = base + (size_t) rows[i] * bpr;
        std::vector<float> v; deq_q8_0_row(row, 160, v);
        double norm = 0; for (float x : v) norm += x * x; norm = sqrt(norm);
        std::vector<uint8_t> orig(bpr);
        fseek(disk, (long)(w->offs + (uint64_t) rows[i] * bpr), SEEK_SET);
        fread(orig.data(), 1, bpr, disk);
        bool differs = memcmp(orig.data(), row, bpr) != 0;
        changed += differs;
        printf("  row %lld: patched_norm=%.4f differs_from_disk=%s first5=[%.3f %.3f %.3f %.3f %.3f]\n",
               (long long) rows[i], norm, differs ? "yes" : "NO", v[0], v[1], v[2], v[3], v[4]);
    }
    fclose(disk);

    if (in_ov == (size_t) n_heads && changed > 0) {
        printf("SEMANTIC OVERLAY TEST PASSED (runtime hash -> patched rows carry injected vectors)\n");
        return 0;
    }
    printf("SEMANTIC TEST FAILED\n");
    return 6;
}
