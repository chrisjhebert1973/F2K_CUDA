# Chapter 28 — Serving: The Persistent Worker & Live Preview

> *Goal of this chapter:* turn the one-shot `generate.cu` of Chapter 23 into something
> you can actually *use* — a **persistent worker** that loads the models once and serves
> jobs over a socket, and a small **web UI** (reachable from a phone over Tailscale) that
> streams the image **forming live** on screen. The interesting engineering is in three
> places: what stays resident, a line-oriented protocol, and decoding low-resolution
> previews mid-denoise without disturbing the run. Anchored to `tools/serve.cu`,
> `tools/webui/app.py`.
>
> *Prerequisites:* Chapters 23 (the end-to-end pipeline), 26–27 (the encoder and image
> conditioning the worker also serves).

---

## 28.1 The model-load tax

`generate.cu` is correct but loads every model on every run: the Qwen3 encoder (four
shards), the transformer, the VAE. On the GB10 that load is ~16 s — and the *actual work*
of a 1024px image is ~8 s. Spending 16 s to do 8 s of work, every time, is the dominant cost
for interactive use. The fix is the oldest one in serving: **load once, stay up, serve many.**
`build/serve` does exactly that, and a same-config image drops from ~25 s (one-shot) to
**~8 s** (resident).

## 28.2 What stays resident, and what gets rebuilt

Not everything can stay fixed: the transformer and VAE are sized for a specific
**resolution and precision**. The worker splits state along that seam:

- **Resolution-independent, resident forever:** the Qwen3 encoder and the BPE tokenizer.
  They depend only on the prompt, never on image size or quant mode, so they load once
  (~16 s) and never reload.
- **Keyed by `(res, precision)`:** the transformer, the VAE **decoder and encoder**
  (Chapter 26), and all the device scratch buffers. These are wrapped in a `Pipeline`
  object that `Worker::ensure(res, precision)` builds lazily and **caches**. A request at
  the same `(res, precision)` reuses it; a change rebuilds it — but only the transformer/VAE
  part, ~6 s, because Qwen is untouched.

So the first image at a new size costs ~14 s; every subsequent image at that size is ~8 s.
This is the same arena-and-reuse discipline as the kernel workspaces (Chapter 20), lifted
to whole models.

## 28.3 A line-oriented protocol

The worker listens on **localhost TCP** and speaks **newline-delimited JSON** — one JSON
object per line, the simplest framing that works. A request is a single line:

```json
{"prompt":"…","res":1024,"precision":"fp8","steps":8,"seed":777,"out":"/abs/path.png",
 "init_image":"…","mask_image":"…","strength":0.6,"stream":false}
```

`init_image` / `mask_image` / `strength` opt into the image-conditioning paths of Chapter 27;
omit them for txt2img. The non-streaming reply is one line:

```json
{"ok":true,"elapsed":8.1,"nans":0,"timing":["text encode 277ms","denoise 6.6s (8 steps)","vae 151ms"]}
```

Binding to `127.0.0.1` keeps the GPU server off the network entirely — the only thing
exposed to other devices is the web front-end (§28.4), which is a separate, authenticated
process. The worker does one job at a time (a single accept loop); concurrency is the web
layer's problem, not the GPU's.

## 28.4 The web front-end

`webui/app.py` is a small Flask app meant to be reached over **Tailscale** from a phone or
tablet — single-user session login, dark mobile UI, no public exposure. It shells nothing
out on the hot path: it opens the worker socket and relays. Three details matter:

- **Batch-aware run metadata.** A "run" is one JSON sidecar holding a *list* of images,
  `images:[{seed,file,ok,nans,timing}]`, so a batch of 4 is one run with four entries. Older
  single-image runs are normalised on read, so the schema change is backward-compatible.
- **One GPU at a time.** A `threading.Lock` serialises GPU jobs across web requests; the
  worker's single accept loop is the second line of defence.
- **Graceful degradation.** If the worker is down, the app falls back to spawning the
  one-shot `generate` binary (slower, but it works), and records which mode was used.

The same endpoints serve txt2img, remix (img2img), inpaint, upscale, and photo upload — all
of which are just different request payloads to the *same* worker (Chapter 27).

## 28.5 Streaming the image as it forms

The feature that makes the whole thing feel alive: while the denoise runs, the browser shows
the image **forming**. The mechanism has three layers.

**Worker side.** When a request sets `"stream":true`, the denoise loop decodes a preview every
~`n_run/4` steps and emits it as a progress line:

```json
{"event":"progress","step":4,"total":8,"w":384,"h":384,"img_b64":"…"}
```

then the normal final line. The preview is produced by `preview_b64`, which decodes the
**in-progress** latent without disturbing the run. The trick is buffer hygiene: it reuses the
scratch buffers that are *free between denoise steps* (`d_velocity`, `d_latent`, `d_pixels`)
and **never touches `d_tokens`** (the live latent). It runs the same bn-denorm → unpatchify →
VAE-decode tail as the final image, then downsamples to 384px and encodes **JPEG q82** —
~40 KB, versus ~500 KB for PNG, a 10× saving that matters over a phone link. Previews are
spaced (≈4 per run) because each is a full VAE decode; spacing keeps the overhead small.

**Relay.** Flask's `/generate_stream` is a streaming `application/x-ndjson` response: a
generator that holds the GPU lock, sends `stream:true` to the worker, and **re-emits each
worker line** to the browser, tagging an `img`/`nimg` index so a batch can report "image 2/4".
On the final line it saves the run metadata and emits `{"event":"complete","rid":…}`.

**Browser.** `streamSubmit` POSTs the form with `fetch`, reads the response with a
`ReadableStream` reader, splits on newlines, and on each `progress` event sets an `<img>` to
`data:image/jpeg;base64,…`. On `complete` it navigates to the result page. No EventSource, no
WebSocket — plain `fetch` streaming, which modern Safari supports.

There is a sharp edge here worth flagging (Appendix B.16): `streamSubmit` is `async`, so it
returns a **Promise**. Wiring it as `onsubmit="return streamSubmit(this)"` returns a *truthy*
value, which does **not** cancel the form's native submit — the browser then POSTs to the
endpoint itself and *downloads the raw NDJSON file*. The fix is to cancel default
synchronously (`onsubmit="streamSubmit(this); return false"`). A reminder that `async`
functions and `onsubmit`'s truthiness contract do not mix.

## 28.6 Robustness: don't let a client kill the GPU

Streaming introduced a failure mode one-shot serving never had: the worker now **writes to
the client repeatedly during a job**, so if the browser disconnects mid-stream, the next
`write` hits a closed socket and the OS raises **`SIGPIPE`** — whose default action is to
*terminate the process*. A user closing a tab was killing the GPU worker (observed as exit
141). One line fixes it — `signal(SIGPIPE, SIG_IGN)` — after which a dead client just makes
`write` return `EPIPE`, the job finishes, and the worker keeps serving (Appendix B.17). The
general lesson: any long-lived server that streams to clients must ignore `SIGPIPE`.

## 28.7 Where this lives in the code

| Concept | Code |
|---|---|
| persistent worker | `tools/serve.cu` (`Worker`, `Pipeline`) |
| resident vs rebuilt state | `Worker::ensure(res, precision)` |
| line protocol | `serve.cu` accept loop (`read_line`, JSON in/out) |
| mid-denoise preview | `Worker::preview_b64` (reuses free buffers, JPEG q82) |
| web UI | `tools/webui/app.py` (Flask, login, batch metadata) |
| streaming relay | `/generate_stream` (NDJSON generator) |
| browser streaming | `streamSubmit` (fetch + ReadableStream) |
| launcher | `tools/webui/start.sh`, `*.service` units |

## 28.8 Summary and what to carry forward

- A **persistent worker** amortises the ~16 s model load: Qwen + tokenizer resident forever,
  transformer + VAE cached by `(res, precision)` and rebuilt only on change. Same-config
  image ~8 s vs ~25 s one-shot.
- The interface is **newline-delimited JSON over localhost TCP**; the GPU server never
  touches the network. The web layer authenticates, serialises the GPU, and degrades to the
  one-shot binary if the worker is down.
- **Live preview** decodes the in-progress latent every ~`n/4` steps into a 384px JPEG,
  reusing the *free* scratch buffers and leaving `d_tokens` untouched, and streams it as
  NDJSON consumed by browser `fetch`/`ReadableStream`.
- Two serving-specific bugs: an `async`-`onsubmit` truthiness trap that **downloaded** the
  stream (B.16), and an unhandled **`SIGPIPE`** that let a disconnecting client kill the
  worker (B.17). Both are one-liners; both only appear once you stream.

That closes **Part VIII** — and the implementation arc of the course. From a text prompt we
now reach a 2K PNG, from an image we reach a restyle or an inpaint, and from a phone we watch
it happen live — all on one Blackwell GPU, in C++/CUDA, validated against diffusers at every
seam. The appendices that follow are the reference layer: the glossary, the **bug museum**
(which the last three chapters fed three fresh entries), and the build-and-reproduce guide.

---

### Exercises

1. **The resident seam.** Justify keeping Qwen resident but rebuilding the transformer on a
   resolution change. What in the transformer's configuration actually depends on `res`, and
   what would it cost to cache *several* `(res, precision)` pipelines at once?
2. **Preview buffer hygiene.** `preview_b64` reuses `d_velocity`/`d_latent`/`d_pixels` but not
   `d_tokens`. Explain why each of those three is safe to clobber *between* denoise steps, and
   what would break if the preview used `d_tokens` as scratch.
3. **Why JPEG, why spaced.** Estimate the bytes streamed for a 16-step run at PNG-every-step
   versus JPEG-every-4-steps, and explain the two independent reasons (encode cost, link
   bandwidth) the implementation chose the latter.
4. **The two streaming bugs.** Both B.16 (download) and B.17 (`SIGPIPE`) are invisible until
   you stream and only appear at a client boundary. Generalise: what class of bug does adding
   a *streaming* interface to a request/response server tend to expose?

*Next: the appendices — [A — Glossary](A-glossary.md), [B — The bug museum](B-bug-museum.md),
[C — Build & reproduce](C-build-reproduce.md).*
