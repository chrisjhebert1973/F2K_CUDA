#include "common/f2k_model_loader.h"
#include "common/tensor.h"
#include <cstdio>
int main() {
    f2k::F2KModelLoader ld;
    for (int i = 1; i <= 4; ++i) {
        char p[256];
        std::snprintf(p, sizeof(p), "/home/chris/models/flux2-klein-9B/qwen3_f2k/shard-%05d.f2k1", i);
        ld.add_shard(p);
    }
    for (const char* n : {"model.layers.0.self_attn.k_proj.weight",
                          "model.layers.0.self_attn.v_proj.weight",
                          "model.layers.0.self_attn.o_proj.weight",
                          "model.layers.0.mlp.down_proj.weight",
                          "model.layers.0.mlp.up_proj.weight"}) {
        const auto* t = ld.find(n);
        if (!t) { std::printf("%s NOT FOUND\n", n); continue; }
        std::printf("%s  dtype=%s  shape=[", n, f2k::dtype_name(t->dtype));
        for (size_t i = 0; i < t->shape.size(); ++i)
            std::printf("%s%lld", i ? "," : "", (long long)t->shape[i]);
        std::printf("]\n");
    }
}
