// Validate the native C++ BPE tokenizer bit-for-bit against the Python
// Qwen2 tokenizer reference (captured via apply_chat_template + tok()).

#include "common/bpe_tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

const char* kTokDir = "/home/chris/models/flux2-klein-9B/tokenizer";

struct Case {
    std::string prompt;
    std::vector<int32_t> expect;  // un-padded chat-templated token ids
};

bool run(const f2k::BpeTokenizer& tok, const Case& c) {
    // seq_len == expect.size() ⇒ no padding, so this is the raw tokenization.
    auto got = tok.encode_for_flux(c.prompt, static_cast<int>(c.expect.size()));
    bool ok = (got == c.expect);
    std::printf("[%s] \"%.40s\"  (%zu ids)\n", ok ? "PASS" : "FAIL",
                c.prompt.c_str(), got.size());
    if (!ok) {
        std::printf("  expect:");
        for (int v : c.expect) std::printf(" %d", v);
        std::printf("\n  got:   ");
        for (int v : got) std::printf(" %d", v);
        std::printf("\n");
    }
    return ok;
}

}  // namespace

int main() {
    f2k::BpeTokenizer tok;
    if (!tok.load(kTokDir)) {
        std::printf("FAIL load: %s\n", tok.error().c_str());
        return 1;
    }

    std::vector<Case> cases = {
        {"a cat on a skateboard",
         {151644, 872, 198, 64, 8251, 389, 264, 97982,
          151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271}},
        {"My dog Rocket, who is an Akita, eating burgers at a really cool biker bar in rural Oklahoma",
         {151644, 872, 198, 5050, 5562, 39218, 11, 879, 374, 458, 16358, 6255, 11,
          12182, 62352, 518, 264, 2167, 7010, 293, 24803, 3619, 304, 19082, 22797,
          151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271}},
        {"It's 3 cats & 2 dogs (100% good) \xE2\x80\x94 don't stop!",
         {151644, 872, 198, 2132, 594, 220, 18, 19423, 609, 220, 17, 12590, 320,
          16, 15, 15, 4, 1661, 8, 1959, 1513, 944, 2936, 0,
          151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271}},
    };

    bool all = true;
    for (const auto& c : cases) all &= run(tok, c);

    // Padding check: pad to 40 with pad id.
    auto padded = tok.encode_for_flux("a cat on a skateboard", 40);
    bool pad_ok = padded.size() == 40 && padded[17] == tok.pad_id() && padded[39] == tok.pad_id();
    std::printf("[%s] padding to 40 (pad id %d)\n", pad_ok ? "PASS" : "FAIL", tok.pad_id());
    all &= pad_ok;

    std::printf("%s\n", all ? "ALL OK" : "FAILURES");
    return all ? 0 : 1;
}
