#include "common/bpe_tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>

namespace f2k {
namespace {

using json = nlohmann::json;

// --- UTF-8 helpers ---------------------------------------------------------
std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

// Decode UTF-8 `s` into codepoints, recording the byte offset of each.
void utf8_decode(const std::string& s, std::vector<uint32_t>& cps,
                 std::vector<size_t>& off) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        off.push_back(i);
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp;
        int len;
        if (c < 0x80)        { cp = c;            len = 1; }
        else if (c < 0xE0)   { cp = c & 0x1F;     len = 2; }
        else if (c < 0xF0)   { cp = c & 0x0F;     len = 3; }
        else                 { cp = c & 0x07;     len = 4; }
        for (int k = 1; k < len && i + k < n; ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        cps.push_back(cp);
        i += len;
    }
    off.push_back(n);  // sentinel
}

// --- \p{L}/\p{N}/\s classifiers (ASCII-precise; non-ASCII → letter) --------
inline bool is_letter(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp >= 0x80;
}
inline bool is_number(uint32_t cp) { return cp >= '0' && cp <= '9'; }
inline bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}
inline bool is_nl(uint32_t cp) { return cp == '\n' || cp == '\r'; }

}  // namespace

bool BpeTokenizer::load(const std::string& dir) {
    // byte → unicode (GPT-2 bytes_to_unicode), encoded as UTF-8.
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i)
        byte_encoder_[bs[i]] = utf8_encode(static_cast<uint32_t>(cs[i]));

    // vocab.json: token → id
    {
        std::ifstream f(dir + "/vocab.json");
        if (!f) { err_ = "cannot open vocab.json"; return false; }
        json v; f >> v;
        for (auto it = v.begin(); it != v.end(); ++it)
            vocab_[it.key()] = it.value().get<int>();
    }
    // merges.txt: "lhs rhs" → rank (skip the #version header / blank lines)
    {
        std::ifstream f(dir + "/merges.txt");
        if (!f) { err_ = "cannot open merges.txt"; return false; }
        std::string line;
        int rank = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            merge_rank_[line] = rank++;  // key is exactly "lhs rhs"
        }
    }
    // added_tokens.json: special token → id (sorted longest-first for greedy match)
    {
        std::ifstream f(dir + "/added_tokens.json");
        if (!f) { err_ = "cannot open added_tokens.json"; return false; }
        json a; f >> a;
        for (auto it = a.begin(); it != a.end(); ++it)
            added_.emplace_back(it.key(), it.value().get<int>());
        std::sort(added_.begin(), added_.end(),
                  [](const auto& x, const auto& y) { return x.first.size() > y.first.size(); });
    }
    ok_ = true;
    return true;
}

// Byte-level BPE of a single pretoken (raw UTF-8 bytes) → token ids.
std::vector<int32_t> BpeTokenizer::bpe_segment(const std::string& bytes) const {
    std::vector<std::string> syms;
    syms.reserve(bytes.size());
    for (unsigned char b : bytes) syms.push_back(byte_encoder_[b]);

    while (syms.size() > 1) {
        int best_rank = INT_MAX;
        size_t best_i = 0;
        bool found = false;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
                found = true;
            }
        }
        if (!found) break;
        syms[best_i] += syms[best_i + 1];
        syms.erase(syms.begin() + static_cast<long>(best_i) + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(syms.size());
    for (const auto& s : syms) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) ids.push_back(it->second);
    }
    return ids;
}

// Pretokenize `text` (the GPT-2/Qwen Split regex, alternatives in order) and
// BPE each chunk, appending ids.
void BpeTokenizer::pretokenize_and_bpe(const std::string& text,
                                       std::vector<int32_t>& out) const {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    utf8_decode(text, cp, off);
    const size_t N = cp.size();
    auto emit = [&](size_t a, size_t b) {  // codepoint range [a,b)
        auto r = bpe_segment(text.substr(off[a], off[b] - off[a]));
        out.insert(out.end(), r.begin(), r.end());
    };

    size_t i = 0;
    while (i < N) {
        // alt1: contractions  (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp[i] == '\'') {
            auto low = [&](size_t j) -> uint32_t {
                if (j >= N) return 0;
                uint32_t c = cp[j];
                return (c >= 'A' && c <= 'Z') ? c + 32 : c;
            };
            uint32_t c1 = low(i + 1), c2 = low(i + 2);
            int adv = 0;
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
                adv = 3;
            else if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
                adv = 2;
            if (adv) { emit(i, i + adv); i += adv; continue; }
        }

        // alt2: [^\r\n\p{L}\p{N}]? \p{L}+
        {
            size_t j = i;
            bool ok = false;
            if (is_letter(cp[i])) {
                ok = true;
            } else if (!is_nl(cp[i]) && !is_number(cp[i]) &&
                       i + 1 < N && is_letter(cp[i + 1])) {
                ++j;  // optional leading non-(nl/letter/number)
                ok = true;
            }
            if (ok) {
                while (j < N && is_letter(cp[j])) ++j;
                emit(i, j); i = j; continue;
            }
        }

        // alt3: \p{N}  (single digit)
        if (is_number(cp[i])) { emit(i, i + 1); i += 1; continue; }

        // alt4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t j = i;
            if (cp[i] == ' ' && i + 1 < N && !is_space(cp[i + 1]) &&
                !is_letter(cp[i + 1]) && !is_number(cp[i + 1]))
                ++j;  // optional single leading space before punctuation
            size_t punct_start = j;
            while (j < N && !is_space(cp[j]) && !is_letter(cp[j]) && !is_number(cp[j])) ++j;
            if (j > punct_start) {                       // matched ≥1 punct/symbol
                while (j < N && is_nl(cp[j])) ++j;       // trailing [\r\n]*
                emit(i, j); i = j; continue;
            }
        }

        // alt5/6/7: whitespace.
        if (is_space(cp[i])) {
            size_t e = i;
            while (e < N && is_space(cp[e])) ++e;
            size_t last_nl = i;
            bool has_nl = false;
            for (size_t k = i; k < e; ++k)
                if (is_nl(cp[k])) { last_nl = k; has_nl = true; }
            if (has_nl) {                                // alt5: \s*[\r\n]+
                emit(i, last_nl + 1); i = last_nl + 1; continue;
            }
            // alt6 \s+(?!\S): EOS run → all; else leave 1. alt7 \s+: lone space.
            size_t j = (e == N) ? e : (e - 1 > i ? e - 1 : i + 1);
            emit(i, j); i = j; continue;
        }

        emit(i, i + 1); i += 1;  // fallback (unreachable for ASCII)
    }
}

std::vector<int32_t> BpeTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    std::string buf;
    auto flush = [&]() {
        if (!buf.empty()) {
            pretokenize_and_bpe(buf, ids);
            buf.clear();
        }
    };
    size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const auto& [s, id] : added_) {  // longest-first
            if (text.compare(pos, s.size(), s) == 0) {
                flush();
                ids.push_back(id);
                pos += s.size();
                matched = true;
                break;
            }
        }
        if (!matched) { buf.push_back(text[pos]); ++pos; }
    }
    flush();
    return ids;
}

std::vector<int32_t> BpeTokenizer::encode_for_flux(const std::string& prompt, int seq_len) const {
    const std::string t = "<|im_start|>user\n" + prompt +
                          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::vector<int32_t> ids = encode(t);
    if (static_cast<int>(ids.size()) > seq_len) ids.resize(seq_len);
    else ids.resize(seq_len, pad_id_);
    return ids;
}

}  // namespace f2k
