# octane_bridge — wire protocol contract

The producer side's statement of what `tools/octane_bridge.cpp` guarantees. This
file is canonical; if it contradicts the code, the code wins and this is the bug.

**Where this file lives**, for anyone writing a mirror header or following one:

    on spark-f42d   /home/chris/Documents/Development/Cpp/Projects/F2KTest/tools/octane/PROTOCOL.md
    on GitHub       chrisjhebert1973/F2K_CUDA -> tools/octane/PROTOCOL.md

Quote it rooted or as the GitHub path. A bare `F2KTest/...` reads as `~/F2KTest`,
which does not exist -- a mirror header carrying that sends its reader nowhere,
which is worse than a stale mirror, because a stale mirror at least announces
itself. (Not hypothetical: the first mirror header said exactly that, and the
consumer had to fetch from GitHub to re-take it.)

Two consumers depend on it today:

* `tools/octane/roadrunner.cxx` — the SGI Octane client, in this repo.
* `LLMTest/src/flux_client.cpp` on **spark-65c1** — an LLM server that drives
  `GEN` as a model-callable tool. **Out of tree**: changes here cannot be found
  by grepping this repo, which is why the framing is frozen and why
  `tests/test_bridge_protocol.cpp` exists to enforce it.

## Endpoint

    spark-f42d.local:1974        octane_bridge   <- talk to this

Not the worker. `serve` binds `127.0.0.1:8765` — loopback only, unreachable off
the box, and it returns a *file path* on f42d rather than image bytes. The bridge
is the only LAN-facing endpoint and the only one that puts pixels on the wire.
The Flask UI (:5000) is for humans, not tools.

## Verbs

    LIST\n
      -> newline-separated model names, then EOF.

    CAPS\n
      -> request limits as key=value lines, then EOF. See "Limits" below for
         which of those keys you should adopt and which you should not.
         Unknown keys may be added; parse by key and ignore what you do not
         recognise. `protocol` bumps only if an existing key changes meaning.

    GEN <res> <steps> <seed> <count> <cfgx100> <seedVar> <varx100> <model>\n
    <prompt>\n
    <negative>\n

    REMIX    <res> <steps> <seed> <count> <strengthx100> <imgW> <imgH> \
             <cfgx100> <seedVar> <varx100> <model>\n
    <prompt>\n
    <negative>\n
    <imgW*imgH*3 raw RGB bytes>

    OUTPAINT ... identical fields and payload to REMIX. The bridge composites a
             zoom-out canvas plus a feathered border mask and inpaints the border.

`<model>` is a name from LIST, or `-` for the bridge default. `seed < 0` means
the bridge randomises and reports the actual seed in each IMAGE frame.

## Response framing

Each message is `'F','2','K','1'` then an int32 type. **All int32 are network
byte order.**

    PROGRESS(1)  idx, total, permille, len, phase[len]
    IMAGE(2)     idx, w, h, seed, pixels[w*h*3]   RGB, top row first
    DONE(3)      count
    ERROR(4)     len, message[len]

A batch is `PROGRESS* (IMAGE PROGRESS*)* DONE`, or a single `ERROR`.

## Limits: facts vs policy

This split matters, and getting it wrong has already nearly cost a consumer a
regression. `CAPS` reports two different kinds of thing in one list.

### Facts — what the bridge accepts. Adopt these.

    res       256 | 512 | 768 | 1024        CAPS: res
    steps     1..30                         CAPS: steps
    count     1..8                          CAPS: count
    strength  5..100 (x100)                 CAPS: strength, image payloads only
    var       0..100 (x100)                 CAPS: var
    cfg       x100; 100 (= 1.0) disables CFG. A negative prompt needs cfg > 100.

These are properties of this bridge. A caller that keeps its own copy will be
silently wrong the day one changes — query `CAPS` instead. That is what `CAPS`
is for.

### Policy — what the bridge *does* with a request that violates them. Do not adopt.

    res_fallback=512                an out-of-range res becomes this...
                                    ...NOT the nearest legal size
    outpaint_strength_trigger=80    outpaint raises strength below this...
    outpaint_strength_set=90        ...to this

Out-of-range values are **clamped, not rejected**. That is a policy, chosen
because the bridge's original client was an SGI Octane that could do nothing
useful with an error — it must pick something and generate.

**A caller with different constraints may correctly choose differently.** A tool
wrapper that can validate before sending and report the correction to its user
is not obliged to substitute what the bridge would have. `LLMTest` substitutes
**1024** for a bad `res` where the bridge substitutes 512, because 1024 is what
its tool schema advertises as the default — and it is right to. It adopts the
legal *set* from `CAPS` and keeps its own fallback.

Reading `res_fallback` as an instruction rather than a description is the trap:
it looks like exactly the cleanup that adopting `CAPS` is meant to be, and it
silently changes behaviour. Take the facts. Decide your own policy.

## Reported adjustments

When the bridge alters a request it says so, as a `PROGRESS` phase before the
batch starts:

    clamped: res 640->512; steps 99->30

Emitted **only** when something actually changed, so silence is meaningful: no
`clamped:` frame means the request ran as sent. The outpaint strength floor
reports the same way (`clamped: strength 60->90`).

A second phase reports which checkpoint actually ran:

    model: flux2-klein-4B fp8

Once per batch. This matters when `<model>` is `-`: the bridge substitutes its
default and the caller never learns what that was. The precision is the quant
that actually *existed* — if the requested one has no shards, `resolve_model`
falls back to the other.

Order within a batch is `clamped:` → `starting` → `model:` → per-step progress.
**Do not depend on that order; match on the prefix.** The test deliberately does
not assert ordering, so it is free to change.

Note for callers that pre-correct against `CAPS`: `clamped:` then becomes a pure
*drift detector* — it can only fire if your view of the limits and the bridge's
have diverged. Log it loudly rather than folding it into a note. It is
load-bearing precisely when it is unexpected.

## Stability contract

`GEN` and `LIST` framing is **frozen**, and `tests/test_bridge_protocol.cpp`
enforces it — including that an out-of-range request lands on exactly the value
`CAPS` advertises. Run `ctest -R bridge_protocol`. It needs no GPU, no model and
no worker, and takes 0.03s.

Additive changes are allowed and have happened: `OUTPAINT`, the `CAPS` verb, and
the `clamped:` / `model:` phases. None moved an existing field or added a frame
type. If the wire format ever has to change incompatibly, the F2K side owns
telling the consumers first.

## Checking a mirror

Consumers may keep a local copy of this file (LLMTest does, at
`docs/F2K_BRIDGE_CONTRACT.md` on spark-65c1, under a header naming this file and
the commit it was taken at). A mirror is one-directional: re-copy it, never edit
it. An edited mirror is the worst case -- it looks synced and is not.

To check one, anchor on the content's first heading rather than a line count:

    diff <(sed -n '/^# octane_bridge/,$p' <mirror>) tools/octane/PROTOCOL.md

Not `tail -n +N`. That hardcodes the current header length, so the day a mirror
is re-taken under a longer header the check fails on content that is byte-identical
-- and a check that cries wolf once teaches the reader to ignore the one signal
that would have caught real drift. (This is not hypothetical: the `tail` form was
written here first, and failed exactly that way when tested.)

The anchor does depend on this file's title line staying `# octane_bridge`. That
is a far less likely edit than the header changing, but if you rename it, fix this
section in the same commit.

## Operational notes

* `f2k-worker`, `f2k-octane-bridge` and `f2k-web` are systemd units and come back
  automatically on boot — no manual step after a power cycle.
* The watchdog probes the worker with model root `/nonexistent/f2k-watchdog-probe`.
  Errors naming that path in the logs are the health check working, not a fault.
* First call after a cold boot pays a model load (~16 s for the encoder). A 9B
  batch at 1024 legitimately runs into minutes — a quiet socket is not
  necessarily a wedged one.

## Verified

2026-08-21: `CAPS`, `LIST` and both phases exercised against the production
bridge on `:1974`, and `CAPS` confirmed reachable across the LAN from spark-65c1.
`tests/test_bridge_protocol.cpp` was validated by sabotage — hardcoding a clamp
site while `CAPS` kept reporting the constant produced the expected failure.
