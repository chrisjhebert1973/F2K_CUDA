# Chapter 22 — A Native Byte-Level BPE Tokenizer

> *Goal of this chapter:* implement the one remaining non-CUDA piece — a
> from-scratch Qwen2/3 byte-level BPE tokenizer in C++ — so that `generate --prompt
> "…" --out x.png` is a single self-contained call with **no Python**. We cover
> byte-level BPE, the bytes-to-unicode map, the GPT-2/Qwen pretokenizer regex (and
> why it must be hand-rolled), the merge algorithm, special-token splitting, the
> chat template, and the bit-exact validation against HuggingFace. Anchored to
> `src/common/bpe_tokenizer.{h,cpp}` and `tests/test_tokenizer.cpp`.
>
> *Prerequisites:* Chapters 05 (the chat template, capture role), 21 (the encoder
> this feeds).

---

## 22.1 Why bother (the motivation)

Before this, tokenization ran in Python: `tools/encode_prompt.py` (or, worse, the
full `diffusers_prompt_embeds.py` which loaded all 15 GB of Qwen3 in PyTorch just to
produce embeddings — ~98 s with the GPU idle, Chapter 25). The encoder itself is
native CUDA (Chapter 21); the tokenizer was the last Python dependency between a
prompt and a PNG. Porting it makes the binary self-contained:

```
 ./generate --prompt "Rocket on a boat" --res 1024 --precision fp8 --out rocket.png
 #  100% C++/CUDA: BpeTokenizer → Qwen3 encoder → transformer → VAE → PNG
```

Tokenization is ~2 s (mostly process/encoder-independent), versus 98 s for the
PyTorch embeds path — and there is no second runtime to install or keep in sync.

## 22.2 What a byte-level BPE tokenizer is

**BPE** (byte-pair encoding) builds a vocabulary by repeatedly merging the most
frequent adjacent symbol pair, learned offline. At inference, tokenizing means
greedily applying those merges to split text into known subword tokens.
**Byte-level** BPE (GPT-2's variant, which Qwen uses) operates on the raw **UTF-8
bytes** of the text, so *any* string is representable (no unknown tokens) — every
byte maps to a base symbol, and merges build up from there.

The pipeline for a piece of text:

```
 text → NFC normalize → pretokenize (regex) → for each chunk:
        bytes → byte-to-unicode map → BPE merge → vocab lookup → token ids
```

Plus a wrapper that splits out **special tokens** and applies the **chat template**.
The five files needed are in the HF tokenizer dir: `vocab.json` (token→id, 151643
entries), `merges.txt` (151387 ranked pairs), `added_tokens.json` (26 special
tokens), `tokenizer_config.json`, `chat_template.jinja`.

## 22.3 The bytes-to-unicode map

The vocabulary's keys are not raw bytes — they are strings of *printable* Unicode
characters, because GPT-2's BPE was defined over a reversible byte→char map that
avoids control characters and spaces in the vocab. The map (`bytes_to_unicode`):
printable ASCII bytes map to themselves; the rest map to codepoints starting at 256.
So byte `0x20` (space) → `Ġ`, byte `0x0A` (newline) → `Ċ`, etc. `bpe_tokenizer.cpp`
builds the 256-entry table at load:

```c++
// printable ranges map to themselves; others to 256, 257, ...
for (b in 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF) bs.push_back(b), cs.push_back(b);
for (b in 0..255 not already in bs) bs.push_back(b), cs.push_back(256 + n++);
byte_encoder_[bs[i]] = utf8_encode(cs[i]);   // each byte → a UTF-8 string
```

So a pretoken's raw UTF-8 bytes become a sequence of these mapped-char strings (e.g.
" dog" → `Ġ d o g`), which is the space BPE operates in and the space `vocab.json`
keys live in (e.g. `"Ġdog"`).

## 22.4 The pretokenizer regex (and why it's hand-rolled)

Before BPE, text is split into chunks by a fixed regex (the GPT-2/Qwen "Split"
pretokenizer). The pattern is:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
 ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
```

In words, tried in order: contractions (`'s`, `'re`, …); an optional leading
non-letter then a run of letters (" dog"); a **single** digit (so "100" → three
tokens); optional space + a run of punctuation; whitespace ending in newlines;
trailing whitespace; any whitespace.

The trap: this uses `\p{L}` (Unicode letter) and `\p{N}` (Unicode number), which
**C++ `std::regex` does not support**. Options were a heavy ICU/Unicode-regex
dependency or a hand-rolled scanner. The project **hand-rolls** it
(`pretokenize_and_bpe`): decode the text to codepoints (tracking byte offsets), and
at each position try the seven alternatives *in order*, classifying each codepoint
with **ASCII-precise** predicates — `is_letter` = `[a-zA-Z]` or any non-ASCII
codepoint, `is_number` = `[0-9]`, `is_space`, `is_nl`. This is **bit-exact for
English prompts** (and very close otherwise — e.g. an em-dash classifies as a
"letter," but because it is space-delimited it still chunks identically, as the
validation confirms). The seven alternatives map directly to seven branches in the
scanner; getting their *order* and the greedy/optional semantics right is the whole
job.

## 22.5 The BPE merge

For each pretoken chunk, BPE merges its mapped-char symbols using the ranked merges:

```c++
// symbols = one mapped-char string per byte of the chunk
while (symbols.size() > 1) {
    find the adjacent pair (a,b) with the LOWEST merge rank   // merge_rank_["a b"]
    if none is mergeable: break
    merge it (a,b) → "ab", removing the second
}
for each remaining symbol: ids.push_back(vocab_[symbol]);
```

Lowest rank = earliest-learned = most frequent merge. Iterate until no adjacent pair
appears in `merges.txt`. Every final symbol is guaranteed to be in `vocab.json`
(base bytes are all there), so the lookups never miss. The merges are loaded from
`merges.txt` keyed by `"left right"` (the `#version` header line is skipped).

## 22.6 Special tokens and the chat template

The chat template (Chapter 05) wraps the prompt:

```
<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

The special tokens (`<|im_start|>`=151644, `<|im_end|>`=151645, `<think>`=151667,
`</think>`=151668, …) are **single ids** that must be matched *literally* and emitted
directly, while the gaps between them are byte-level-BPE'd. `encode()` scans the
string, at each position checking the 26 added tokens **longest-first**; on a match it
flushes the pending text buffer (pretokenize+BPE) and emits the special id; otherwise
it accumulates bytes. `encode_for_flux()` builds the template, calls `encode()`, and
**right-pads/truncates to 512** with the pad token (`<|endoftext|>`=151643) — exactly
matching `tools/encode_prompt.py` and the encoder's expected input (Chapter 21).

(NFC normalization is specified by the tokenizer but is a no-op for ASCII prompts;
it is noted as a known limitation for exotic inputs.)

## 22.7 Bit-exact validation

A tokenizer that is "almost right" silently shifts conditioning. The defense is a
**bit-for-bit** comparison against the Python HF tokenizer
(`tests/test_tokenizer.cpp`, ctest `tokenizer`): for several prompts, check that
`encode_for_flux(prompt, N)` equals the reference id vector exactly. The reference
cases deliberately exercise the hard parts:

```
 "a cat on a skateboard"                          → 17 ids  PASS
 "My dog Rocket, who is an Akita, ... Oklahoma"    → 34 ids  PASS
 "It's 3 cats & 2 dogs (100% good) — don't stop!"  → 33 ids  PASS   (em-dash, contractions,
                                                                     single-digit splits, '&', '!')
 + a padding check (pad to 40 with id 151643)               PASS
 ALL OK
```

The third case is the stress test: the em-dash (multi-byte UTF-8), the `'t`/`'don`
contraction handling, single-digit tokenization of "100" (→ `1`,`0`,`0`), and `&`/`!`
punctuation — all reproduced exactly. Passing these (and the real Rocket prompts)
means the native tokenizer is interchangeable with HF's for this model.

## 22.8 The payoff, end to end

With the tokenizer native, the single-call path works (Chapter 23): tokenize (~2 s,
no model load) → Qwen3 encoder (Chapter 21) → transformer → VAE → **PNG via stb**.
The slow PyTorch embeds path (98 s, 15 GB, GPU idle) is now optional/legacy; the
images are indistinguishable (the encoder is the same NVFP4 Qwen3 either way — the
~8% pixel difference vs the diffusers-embeds path is the NVFP4 encoder vs FP32, a
valid alternate composition, Chapter 25). The only non-CUDA logic left in the hot
path is this pure-C++ tokenizer, and it is ~2 s of mostly-fixed cost.

## 22.9 Where this lives in the code

| Concept | Code |
|---|---|
| the tokenizer | `src/common/bpe_tokenizer.{h,cpp}` (`BpeTokenizer`) |
| bytes-to-unicode, vocab/merges load | `BpeTokenizer::load` |
| hand-rolled pretokenizer | `BpeTokenizer::pretokenize_and_bpe` |
| BPE merge | `BpeTokenizer::bpe_segment` |
| special-token split + template | `BpeTokenizer::encode`, `encode_for_flux` |
| validation | `tests/test_tokenizer.cpp` (ctest `tokenizer`) |
| reference dumper | the Python HF tokenizer (one-off, to produce ref vectors) |
| in-pipeline use | `generate.cu` `--prompt` path |

## 22.10 Summary and what to carry forward

- A **byte-level BPE** tokenizer makes any string representable; tokenizing is:
  NFC → **regex pretokenize** → **bytes-to-unicode** → **BPE merge** → vocab lookup,
  wrapped by special-token splitting and the chat template.
- The pretokenizer regex uses `\p{L}/\p{N}` which `std::regex` lacks, so it is
  **hand-rolled** with ASCII-precise classifiers — bit-exact for English prompts.
- BPE merges by repeatedly combining the **lowest-rank** adjacent pair; special
  tokens are matched **longest-first** and emitted as single ids; the output is
  **right-padded to 512** with `<|endoftext|>`.
- Validated **bit-for-bit** against HF on contraction/digit/punctuation/em-dash
  cases, making it a drop-in for the Python tokenizer and completing the
  **self-contained** `--prompt → PNG` path.

That closes Part VI. Part VII integrates everything: the end-to-end pipeline
(Chapter 23), the performance methodology that found every bottleneck (Chapter 24),
and the validation methodology that made the port trustworthy (Chapter 25).

---

### Exercises

1. **Byte-level coverage.** Explain why byte-level BPE never needs an `<unk>` token,
   and what the bytes-to-unicode map buys over operating on raw bytes directly.
2. **Pretokenizer order.** For `" 2"` (space then digit) and `" dog"` (space then
   letters), trace which of the seven alternatives fires and why the leading space
   ends up attached (or not). Tie to the reference ids (` 2`→ space+`2`; ` dog`→ one
   token).
3. **Why bit-exact matters.** Argue why an "almost correct" tokenizer is more
   dangerous than an obviously broken one, with reference to Chapter 05's
   conditioning role. What would a 1-token shift do?
4. **The hand-roll trade.** What does the ASCII-precise pretokenizer get wrong for
   non-ASCII punctuation, and why does the em-dash test still pass? When would you
   need a full Unicode-property implementation?

*Next: [Chapter 23 — The end-to-end pipeline](23-end-to-end.md), opening Part VII.*
