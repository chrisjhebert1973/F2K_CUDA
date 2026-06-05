// MMDiTStack — chains N MMDiTBlocks, in-place on x, sharing one workspace.

#include "backend/cuda/mmdit_stack.h"
#include "backend/cuda/mmdit_block.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace f2k::cuda {

struct MMDiTStack::Impl {
    int   num_blocks = 0;
    bool  valid = false;
    std::string err;
    std::vector<std::unique_ptr<MMDiTBlock>> blocks;
    size_t per_block_ws = 0;   // max across all blocks (they should all match)
};

MMDiTStack::MMDiTStack(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    auto fail = [&](std::string s) { I.err = std::move(s); };

    if (cfg.num_blocks <= 0) { fail("MMDiTStack: num_blocks must be > 0"); return; }
    if (!cfg.W_norm1 || !cfg.W_q || !cfg.W_k || !cfg.W_v || !cfg.W_proj ||
        !cfg.W_norm2 || !cfg.W_gate || !cfg.W_up || !cfg.W_down) {
        fail("MMDiTStack: a weight pointer array is null"); return;
    }

    I.num_blocks = cfg.num_blocks;
    I.blocks.reserve(cfg.num_blocks);

    for (int b = 0; b < cfg.num_blocks; ++b) {
        MMDiTBlock::Config bc;
        bc.batch    = cfg.batch;
        bc.seq      = cfg.seq;
        bc.n_heads  = cfg.n_heads;
        bc.head_dim = cfg.head_dim;
        bc.ffn_dim  = cfg.ffn_dim;
        bc.rms_eps  = cfg.rms_eps;
        bc.W_norm1  = cfg.W_norm1[b];
        bc.W_q      = cfg.W_q[b];
        bc.W_k      = cfg.W_k[b];
        bc.W_v      = cfg.W_v[b];
        bc.W_proj   = cfg.W_proj[b];
        bc.W_norm2  = cfg.W_norm2[b];
        bc.W_gate   = cfg.W_gate[b];
        bc.W_up     = cfg.W_up[b];
        bc.W_down   = cfg.W_down[b];

        auto blk = std::make_unique<MMDiTBlock>(bc);
        if (!blk->ok()) {
            fail(std::string("block ") + std::to_string(b) + ": " + blk->last_error());
            return;
        }
        if (blk->workspace_size_bytes() > I.per_block_ws) {
            I.per_block_ws = blk->workspace_size_bytes();
        }
        I.blocks.push_back(std::move(blk));
    }
    I.valid = true;
}

MMDiTStack::~MMDiTStack() = default;

bool        MMDiTStack::ok()                  const { return impl_ && impl_->valid; }
const char* MMDiTStack::last_error()          const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
int    MMDiTStack::num_blocks()              const { return impl_ ? impl_->num_blocks : 0; }
size_t MMDiTStack::workspace_size_bytes()    const { return impl_ ? impl_->per_block_ws : 0; }

bool MMDiTStack::forward(void* x_bf16,
                         const Modulation& mod,
                         const float* rope_cos, const float* rope_sin,
                         void* workspace, size_t workspace_size,
                         cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (workspace_size < I.per_block_ws) {
        I.err = "MMDiTStack: workspace too small for per-block requirement";
        return false;
    }

    for (int b = 0; b < I.num_blocks; ++b) {
        MMDiTBlock::Modulation bm;
        bm.scale_attn = mod.scale_attn ? mod.scale_attn[b] : nullptr;
        bm.shift_attn = mod.shift_attn ? mod.shift_attn[b] : nullptr;
        bm.gate_attn  = mod.gate_attn  ? mod.gate_attn[b]  : nullptr;
        bm.scale_mlp  = mod.scale_mlp  ? mod.scale_mlp[b]  : nullptr;
        bm.shift_mlp  = mod.shift_mlp  ? mod.shift_mlp[b]  : nullptr;
        bm.gate_mlp   = mod.gate_mlp   ? mod.gate_mlp[b]   : nullptr;

        if (!I.blocks[b]->forward(x_bf16, bm, rope_cos, rope_sin,
                                  workspace, workspace_size, stream)) {
            I.err = std::string("block ") + std::to_string(b) + ": " +
                    I.blocks[b]->last_error();
            return false;
        }
    }
    return true;
}

} // namespace f2k::cuda
