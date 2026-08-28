// End-to-end test of the PLE overlay through the real loader path:
//   llama_model_loader(fname) -> init_mappings -> apply_ple_overlay
// then read patched rows through the mapping and compare to the overlay file.
#include "llama-model-loader.h"
#include "llama-ple-overlay.h"
#include "llama.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

static bool file_eq_region(const std::string & path, uint64_t off, const uint8_t * want, size_t n) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, off, SEEK_SET);
    std::vector<uint8_t> buf(n);
    size_t got = fread(buf.data(), 1, n, f);
    fclose(f);
    return got == n && memcmp(buf.data(), want, n) == 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <overlay.plepatch>\n", argv[0]); return 2; }
    const std::string model = argv[1];
    const std::string ov_path = argv[2];

    // 1) parse overlay index independently
    llama_ple_overlay ov = llama_ple_overlay_read_index(ov_path.c_str());
    printf("overlay: tensor=%s rows=%zu bpr=%llu qtype=%u\n",
           ov.tensor_name.c_str(), ov.rows.size(),
           (unsigned long long) ov.bytes_per_row, ov.qtype);

    // 2) real loader, mmap, no_alloc (metadata + file handles only)
    std::vector<std::string> splits; // empty -> auto-discover
    llama_model_loader ml(nullptr, nullptr, nullptr, model, splits, nullptr,
                          LLAMA_LOAD_MODE_MMAP, /*check_tensors=*/false,
                          /*no_alloc=*/true, /*load_mtp=*/false, nullptr, nullptr);
    printf("loader: n_tensors=%d files=%zu\n", ml.n_tensors, ml.files.size());

    // find the table weight
    const auto * w = ml.get_weight(ov.tensor_name.c_str());
    if (!w) { fprintf(stderr, "table tensor not found\n"); return 3; }
    printf("table weight: file_idx=%u ne=[%lld,%lld] type=%d\n",
           w->idx, (long long) w->tensor->ne[0], (long long) w->tensor->ne[1],
           (int) w->tensor->type);

    // 3) mappings + overlay (the production sequence)
    // lazy mapping (no MAP_POPULATE/WILLNEED): this test touches only the 26
    // patched rows; prefetching would fault in all ~179 GB of shards and OOM
    ml.init_mappings(false, nullptr);
    if (!(argc > 3 && std::string(argv[3]) == "env")) {
        ml.ple_overlay = ov;        // simulate env/sibling discovery
    }
    ml.apply_ple_overlay();

    // 4) read patched rows through the mapping and compare to overlay payloads
    uint8_t * base = (uint8_t *) ml.mappings.at(w->idx)->addr();
    const uint64_t data_off = w->offs;
    const size_t bpr = ov.bytes_per_row;

    // load overlay payloads
    FILE * f = fopen(ov_path.c_str(), "rb");
    fseek(f, 128, SEEK_SET);
    // skip manifest
    uint8_t hdr[128]; fseek(f, 0, SEEK_SET); fread(hdr, 1, 128, f);
    auto rd = [&](int o, int n){ uint64_t v=0; for(int i=0;i<n;i++) v |= (uint64_t)hdr[o+i]<<(8*i); return v; };
    uint64_t mlen = rd(40,8);
    fseek(f, 128 + mlen, SEEK_SET);
    size_t checked = 0, bad = 0;
    for (const auto & r : ov.rows) {
        uint64_t fid = 0;
        if (fread(&fid, 8, 1, f) != 1) { fprintf(stderr, "short id\n"); return 4; }
        if (fid != r.row) { fprintf(stderr, "id order mismatch: file %llu index %llu\n",
                                 (unsigned long long) fid, (unsigned long long) r.row); return 7; }
        std::vector<uint8_t> want(bpr);
        if (fread(want.data(), 1, bpr, f) != bpr) { fprintf(stderr, "short payload\n"); return 4; }
        uint8_t * row = base + data_off + r.row * bpr;
        if (memcmp(row, want.data(), bpr) != 0) {
            bad++;
            if (bad <= 3) printf("MISMATCH row %llu\n", (unsigned long long) r.row);
        }
        checked++;
    }
    fclose(f);
    printf("checked %zu patched rows through the mapping, %zu mismatches\n", checked, bad);

    // 5) verify the file on disk is untouched for one patched row
    //    (re-read from disk; it must NOT equal the overlay payload)
    {
        FILE * g = fopen(model.c_str(), "rb");
        // find the shard file that holds the table
        // ml.file_paths[w->idx] is the shard
        const std::string shard = ml.file_paths.at(w->idx);
        fclose(g);
        // read original bytes at row 0 of the overlay
        uint64_t row0 = ov.rows.front().row;
        std::vector<uint8_t> disk(bpr);
        FILE * h = fopen(shard.c_str(), "rb");
        fseek(h, data_off + row0 * bpr, SEEK_SET);
        fread(disk.data(), 1, bpr, h);
        fclose(h);
        // overlay payload for row0
        f = fopen(ov_path.c_str(), "rb");
        fseek(f, 128 + mlen, SEEK_SET);
        std::vector<uint8_t> pay(bpr);
        for (const auto & r : ov.rows) {
            uint64_t fid = 0; fread(&fid, 8, 1, f);
            if (fid == row0) { fread(pay.data(), 1, bpr, f); break; }
            fseek(f, bpr, SEEK_CUR);
        }
        fclose(f);
        bool disk_untouched = memcmp(disk.data(), pay.data(), bpr) != 0;
        printf("disk bytes at patched row %llu differ from overlay payload: %s (expect 1)\n",
               (unsigned long long) row0, disk_untouched ? "yes" : "NO");
        if (!disk_untouched) return 5;
    }

    if (bad == 0 && checked == ov.rows.size()) {
        printf("LOADER OVERLAY E2E PASSED\n");
        return 0;
    }
    return 6;
}
