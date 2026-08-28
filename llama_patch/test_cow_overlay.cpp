// Direct test of llama_ple_overlay_apply: build a fake table file, mmap it
// read-only, apply an overlay, verify patched rows are visible and the file
// on disk is untouched.
#include "llama-ple-overlay.h"
#include "llama-mmap.h"
#include "llama-io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cassert>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static void write_file(const char * p, const std::vector<uint8_t> & d) {
    FILE * f = fopen(p, "wb"); fwrite(d.data(), 1, d.size(), f); fclose(f);
}

int main() {
    // artifact dir may not exist (e.g. fresh boot wiped /tmp); create it
    if (mkdir("/tmp/ngt", 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "cannot create /tmp/ngt: %s\n", strerror(errno));
        return 3;
    }
    const int64_t n_rows = 1000;
    const int64_t bpr = 170;
    const int64_t data_off = 4096;
    const size_t fsz = data_off + n_rows * bpr;

    // fake model file: every byte = row index mod 251
    std::vector<uint8_t> model(fsz);
    for (int64_t r = 0; r < n_rows; r++)
        for (int64_t b = 0; b < bpr; b++)
            model[data_off + r * bpr + b] = (uint8_t)((r * 7 + b) % 251);
    write_file("/tmp/ngt/fake_model.bin", model);

    // fake overlay: patch rows 5, 6 (same page), 42, 999
    std::vector<uint64_t> rows = {5, 6, 42, 999};
    std::vector<std::vector<uint8_t>> vals;
    for (auto r : rows) {
        std::vector<uint8_t> v(bpr);
        for (int64_t b = 0; b < bpr; b++) v[b] = (uint8_t)(0xC0 ^ (r + b));
        vals.push_back(v);
    }
    {
        FILE * f = fopen("/tmp/ngt/fake.plepatch", "wb");
        uint8_t hdr[128] = {0};
        memcpy(hdr, "PLEOVLY1", 8);
        auto put = [&](int off, uint64_t v, int n){ for(int i=0;i<n;i++) hdr[off+i]=(v>>(8*i))&0xFF; };
        put(8, 1, 2); put(12, 8, 4); put(16, 160, 8); put(24, bpr, 8);
        put(32, rows.size(), 8); put(40, 2, 8); // manifest = "{}"
        memcpy(hdr + 56, "per_layer_token_embd.weight", 27);
        fwrite(hdr, 1, 128, f); fwrite("{}", 1, 2, f);
        for (size_t i = 0; i < rows.size(); i++) {
            fwrite(&rows[i], 8, 1, f);
            fwrite(vals[i].data(), 1, bpr, f);
        }
        fclose(f);
    }

    // open + mmap read-only shared (like llama.cpp)
    int fd = open("/tmp/ngt/fake_model.bin", O_RDONLY);
    void * base = mmap(NULL, fsz, PROT_READ, MAP_SHARED, fd, 0);
    assert(base != MAP_FAILED);

    // sanity: original bytes visible
    assert(((uint8_t*)base)[data_off + 5 * bpr] == (uint8_t)((5 * 7) % 251));

    // apply overlay
    llama_ple_overlay ov = llama_ple_overlay_read_index("/tmp/ngt/fake.plepatch");
    assert(ov.rows.size() == 4);
    assert(ov.tensor_name == "per_layer_token_embd.weight");
    assert(ov.bytes_per_row == (uint64_t)bpr);

    // We need a llama_mmap* for the API; emulate by calling the raw logic via
    // a tiny shim: the apply function only uses mapping->addr().
    // Build a llama_mmap around the file.
    llama_file lf("/tmp/ngt/fake_model.bin", "rb");
    llama_mmap mm(&lf, 0, false, {});
    size_t applied = llama_ple_overlay_apply(&mm, ov, "/tmp/ngt/fake_model.bin", data_off, n_rows);
    printf("applied=%zu\n", applied);
    assert(applied == 4);

    // verify patched rows visible through the ORIGINAL mapping addr
    uint8_t * m = (uint8_t*)mm.addr();
    for (size_t i = 0; i < rows.size(); i++) {
        for (int64_t b = 0; b < bpr; b++) {
            if (m[data_off + rows[i] * bpr + b] != vals[i][b]) {
                printf("MISMATCH row %llu byte %lld got %02x want %02x\n",
                       (unsigned long long)rows[i], (long long)b,
                       m[data_off + rows[i] * bpr + b], vals[i][b]);
                return 2;
            }
        }
    }
    printf("OK: all 4 patched rows visible through read-only mapping\n");

    // verify an UNPATCHED row is untouched
    for (int64_t b = 0; b < bpr; b++)
        assert(m[data_off + 500 * bpr + b] == (uint8_t)((500 * 7 + b) % 251));
    printf("OK: unpatched row 500 intact\n");

    // verify file on disk is byte-identical to original
    std::vector<uint8_t> disk(fsz);
    int fd2 = open("/tmp/ngt/fake_model.bin", O_RDONLY);
    assert(read(fd2, disk.data(), fsz) == (ssize_t)fsz);
    assert(disk == model);
    printf("OK: file on disk byte-identical (COW, no writes)\n");

    close(fd2); close(fd); munmap(base, fsz);
    printf("ALL COW OVERLAY TESTS PASSED\n");
    return 0;
}
