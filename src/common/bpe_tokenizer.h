// Native Qwen2/3 byte-level BPE tokenizer (no Python).
//
// Reproduces the HF `Qwen2TokenizerFast` pipeline used by FLUX.2-klein's text
// encoder: NFC (ASCII no-op) → GPT-2/Qwen Split regex → ByteLevel byte→unicode
// map → BPE merge → vocab lookup, plus the Qwen3 chat template and special-token
// splitting. `encode_for_flux` applies the user-message chat template
// (add_generation_prompt, thinking disabled), tokenizes, and right-pads/truncates
// to a fixed length with the pad token — matching tools/encode_prompt.py exactly.
//
// The regex's \p{L}/\p{N} classes are handled with an ASCII-precise classifier
// (non-ASCII codepoints treated as letters) — bit-exact for English prompts.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace f2k {

class BpeTokenizer {
public:
    // Loads vocab.json, merges.txt, added_tokens.json from a HF tokenizer dir.
    bool load(const std::string& tokenizer_dir);

    // Chat-template + tokenize a single user prompt, right-pad/truncate to
    // seq_len with the pad token. Returns exactly seq_len ids.
    std::vector<int32_t> encode_for_flux(const std::string& prompt, int seq_len) const;

    // Raw tokenize of arbitrary text (special tokens recognized, no template).
    std::vector<int32_t> encode(const std::string& text) const;

    bool ok() const { return ok_; }
    const std::string& error() const { return err_; }
    int pad_id() const { return pad_id_; }

private:
    std::vector<int32_t> bpe_segment(const std::string& bytes) const;
    void pretokenize_and_bpe(const std::string& text, std::vector<int32_t>& out) const;

    std::unordered_map<std::string, int> vocab_;       // mapped-char token → id
    std::unordered_map<std::string, int> merge_rank_;  // "lhs rhs" → rank
    std::string byte_encoder_[256];                    // byte → UTF-8 of mapped cp
    // Added/special tokens, sorted by descending length for greedy matching.
    std::vector<std::pair<std::string, int>> added_;
    int pad_id_ = 151643;                              // <|endoftext|>
    bool ok_ = false;
    std::string err_;
};

}  // namespace f2k
