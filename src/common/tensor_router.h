// TensorRouter — maps F2K1 tensor names to structural roles in the FLUX.2-klein
// MMDiT transformer. Pure metadata layer over F2KModelLoader; doesn't move
// data. Used to translate "model file with 233 named blobs" into "block i has
// these specific weight tensors."
//
// Tensor name patterns this router understands (all suffixed by `.weight`):
//
//   Global:
//     x_embedder, context_embedder, proj_out
//     time_guidance_embed.timestep_embedder.linear_{1,2}
//     double_stream_modulation_{img,txt}.linear
//     single_stream_modulation.linear
//     norm_out.linear
//
//   Per double-stream block (transformer_blocks.{i}.*):
//     attn.{norm_q,norm_k,norm_added_q,norm_added_k}           (rank-1 gains)
//     attn.{to_q,to_k,to_v}                                    (image QKV)
//     attn.to_out.0                                            (image proj)
//     attn.{add_q_proj,add_k_proj,add_v_proj}                  (text QKV)
//     attn.to_add_out                                          (text proj)
//     ff.linear_{in,out}                                       (image GeGLU MLP)
//     ff_context.linear_{in,out}                               (text GeGLU MLP)
//
//   Per single-stream block (single_transformer_blocks.{i}.*):
//     attn.{norm_q,norm_k}
//     attn.to_qkv_mlp_proj                                     (fused QKV+MLP_in)
//     attn.to_out                                              (fused proj+MLP_down)
//
// Modulation note: there are NO per-block modulation tensors. The three
// {double_img, double_txt, single}_modulation.linear blobs are SHARED across
// all blocks of that stream type — this is a distilled-model design choice.

#pragma once

#include "common/f2k_model_loader.h"
#include "common/tensor.h"

#include <string>
#include <vector>

namespace f2k {

struct DoubleBlockTensors {
    // Image stream
    const TensorView* norm_q          = nullptr;
    const TensorView* norm_k          = nullptr;
    const TensorView* to_q            = nullptr;
    const TensorView* to_k            = nullptr;
    const TensorView* to_v            = nullptr;
    const TensorView* to_out          = nullptr;    // attn.to_out.0.weight
    const TensorView* ff_in           = nullptr;    // ff.linear_in.weight
    const TensorView* ff_out          = nullptr;    // ff.linear_out.weight

    // Text stream
    const TensorView* norm_added_q    = nullptr;
    const TensorView* norm_added_k    = nullptr;
    const TensorView* add_q_proj      = nullptr;
    const TensorView* add_k_proj      = nullptr;
    const TensorView* add_v_proj      = nullptr;
    const TensorView* to_add_out      = nullptr;
    const TensorView* ff_ctx_in       = nullptr;    // ff_context.linear_in.weight
    const TensorView* ff_ctx_out      = nullptr;
};

struct SingleBlockTensors {
    const TensorView* norm_q          = nullptr;
    const TensorView* norm_k          = nullptr;
    const TensorView* to_qkv_mlp_proj = nullptr;
    const TensorView* to_out          = nullptr;
};

struct GlobalTensors {
    const TensorView* x_embedder         = nullptr;
    const TensorView* context_embedder   = nullptr;
    const TensorView* time_embed_1       = nullptr;
    const TensorView* time_embed_2       = nullptr;
    const TensorView* double_mod_img     = nullptr;
    const TensorView* double_mod_txt     = nullptr;
    const TensorView* single_mod         = nullptr;
    const TensorView* norm_out           = nullptr;
    const TensorView* proj_out           = nullptr;
};

class TensorRouter {
public:
    explicit TensorRouter(const F2KModelLoader& loader);

    // Parses every tensor name; on success populates all the structures
    // below. Returns false if any required tensor is missing or the same
    // role is claimed by two tensors.
    bool build();

    const GlobalTensors&                    globals()       const { return globals_; }
    const std::vector<DoubleBlockTensors>&  double_blocks() const { return double_blocks_; }
    const std::vector<SingleBlockTensors>&  single_blocks() const { return single_blocks_; }

    int num_double_blocks() const { return static_cast<int>(double_blocks_.size()); }
    int num_single_blocks() const { return static_cast<int>(single_blocks_.size()); }

    // Tensors in the loader that we couldn't classify (e.g., bn.* metadata).
    const std::vector<std::string>& unrouted() const { return unrouted_; }

    const std::string& last_error() const { return last_error_; }

private:
    const F2KModelLoader&            loader_;
    GlobalTensors                    globals_;
    std::vector<DoubleBlockTensors>  double_blocks_;
    std::vector<SingleBlockTensors>  single_blocks_;
    std::vector<std::string>         unrouted_;
    std::string                      last_error_;
};

} // namespace f2k
