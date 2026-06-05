// Synthesizes a tiny safetensors file, converts it to F2K, reads back, and
// verifies bit-identical round-trip. Pure host code, no GPU, no real model.

#include "common/f2k_format.h"
#include "common/safetensors.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace f2k;

namespace {

// Synthesize a safetensors file containing two small tensors:
//   "a": F16   shape [4, 2] = 8 elements × 2 bytes = 16 bytes
//   "b": BF16  shape [2, 2] = 4 elements × 2 bytes = 8  bytes
// Then a third with U8 to exercise byte-typed handling.
void write_synth(const std::string& path) {
    // Note: HF's spec mandates JSON keys appear in declaration order in the
    // header, and the data must appear at the byte offsets listed.
    const std::string json =
        R"({"a":{"dtype":"F16","shape":[4,2],"data_offsets":[0,16]},)"
        R"("b":{"dtype":"BF16","shape":[2,2],"data_offsets":[16,24]},)"
        R"("c":{"dtype":"U8","shape":[8],"data_offsets":[24,32]}})";

    const uint64_t jlen = json.size();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&jlen), 8);
    f.write(json.data(), static_cast<std::streamsize>(jlen));

    // 32 bytes of synthetic data: 0x10..0x2f
    uint8_t blob[32];
    for (int i = 0; i < 32; ++i) blob[i] = static_cast<uint8_t>(0x10 + i);
    f.write(reinterpret_cast<const char*>(blob), sizeof(blob));
}

bool compare(const TensorView* s, const TensorView* d, const char* tag) {
    if (!d) {
        std::fprintf(stderr, "FAIL [%s]: missing in destination\n", tag);
        return false;
    }
    if (s->dtype != d->dtype) {
        std::fprintf(stderr, "FAIL [%s]: dtype %s vs %s\n", tag,
                     dtype_name(s->dtype), dtype_name(d->dtype));
        return false;
    }
    if (s->shape != d->shape) {
        std::fprintf(stderr, "FAIL [%s]: shape mismatch\n", tag);
        return false;
    }
    if (s->data_size != d->data_size) {
        std::fprintf(stderr, "FAIL [%s]: data_size %zu vs %zu\n", tag,
                     s->data_size, d->data_size);
        return false;
    }
    if (std::memcmp(s->data, d->data, s->data_size) != 0) {
        std::fprintf(stderr, "FAIL [%s]: byte-for-byte mismatch\n", tag);
        return false;
    }
    return true;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "f2k_test_round_trip";
    fs::create_directories(tmp);
    const std::string st_path = (tmp / "synth.safetensors").string();
    const std::string fk_path = (tmp / "synth.f2k1").string();

    write_synth(st_path);
    std::printf("[1/4] wrote %s\n", st_path.c_str());

    Safetensors st;
    if (!st.open(st_path)) {
        std::fprintf(stderr, "safetensors open failed: %s\n", st.last_error().c_str());
        return 1;
    }
    std::printf("[2/4] safetensors opened: %zu tensors\n", st.names().size());

    {
        F2KWriter w(fk_path);
        if (!w.ok()) {
            std::fprintf(stderr, "F2KWriter ctor failed: %s\n", w.last_error().c_str());
            return 1;
        }
        for (const auto& name : st.names()) {
            const TensorView* v = st.find(name);
            if (!w.add_tensor(name, v->dtype, v->shape, v->data, v->data_size)) {
                std::fprintf(stderr, "add_tensor(%s): %s\n",
                             name.c_str(), w.last_error().c_str());
                return 1;
            }
        }
        if (!w.commit()) {
            std::fprintf(stderr, "commit: %s\n", w.last_error().c_str());
            return 1;
        }
    }
    std::printf("[3/4] wrote %s\n", fk_path.c_str());

    F2KReader r;
    if (!r.open(fk_path)) {
        std::fprintf(stderr, "F2KReader open failed: %s\n", r.last_error().c_str());
        return 1;
    }
    std::printf("[4/4] F2K opened: %llu tensors, manifest %llu bytes, data %llu bytes\n",
                static_cast<unsigned long long>(r.header().total_tensors),
                static_cast<unsigned long long>(r.header().manifest_size_bytes),
                static_cast<unsigned long long>(r.header().data_size_bytes));

    bool ok = true;
    for (const auto& name : st.names()) {
        ok &= compare(st.find(name), r.find(name), name.c_str());
    }
    if (ok) std::printf("ROUND-TRIP OK\n");
    return ok ? 0 : 1;
}
