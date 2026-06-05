// Token-embedding lookup. Gathers per-token rows from a [vocab, hidden] table
// into [seq, hidden]. BF16 storage; row-pointer copy.

#pragma once

#include <cstdint>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// table:    [vocab, hidden]  BF16 device
// token_ids: [seq]           int32 device
// out:       [seq, hidden]   BF16 device
bool embed_lookup_bf16(const void* table_bf16,
                       const int32_t* token_ids,
                       void* out_bf16,
                       int seq, int hidden, int vocab,
                       cudaStream_t stream = nullptr);

} // namespace f2k::cuda
