#include "common/f2k_format.h"
#include "common/tensor.h"
#include <cstdio>

int main() {
    f2k::F2KReader r;
    if (!r.open("/home/chris/models/flux2-klein-9B/vae_f2k/vae.f2k1")) {
        std::printf("open failed\n"); return 1;
    }
    for (const auto& name : r.names()) {
        const auto* t = r.find(name);
        if (!t) continue;
        std::printf("%s  dtype=%s  shape=[", name.c_str(), f2k::dtype_name(t->dtype));
        for (size_t i = 0; i < t->shape.size(); ++i) {
            std::printf("%s%lld", i ? "," : "", (long long)t->shape[i]);
        }
        std::printf("]\n");
    }
    return 0;
}
