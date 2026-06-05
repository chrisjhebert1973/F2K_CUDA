#include "common/f2k_model_loader.h"
#include "common/tensor.h"
#include <cstdio>
int main() {
    f2k::F2KModelLoader ld;
    ld.add_shard("/home/chris/models/flux2-klein-9B/transformer_f2k/shard-00001.f2k1");
    ld.add_shard("/home/chris/models/flux2-klein-9B/transformer_f2k/shard-00002.f2k1");
    for (const char* n : {"x_embedder.weight", "proj_out.weight"}) {
        const auto* t = ld.find(n);
        if (!t) { std::printf("%s NOT FOUND\n", n); continue; }
        std::printf("%s  dtype=%s  shape=[", n, f2k::dtype_name(t->dtype));
        for (size_t i = 0; i < t->shape.size(); ++i)
            std::printf("%s%lld", i ? "," : "", (long long)t->shape[i]);
        std::printf("]\n");
    }
}
