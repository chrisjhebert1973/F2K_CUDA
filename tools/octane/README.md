# Roadrunner — SGI Octane → DGX Spark image generator

A ViewKit/Motif front-end that runs on an SGI Octane (IRIX / MIPSpro) and drives
the F2K_CUDA FLUX.2-klein image generator on the Spark over the LAN. Type a
prompt on a 1990s workstation, watch a Blackwell GPU paint it back.

```
  Octane "Roadrunner"                         Spark "sparky"
  ┌────────────────────┐                      ┌────────────────────────────┐
  │ roadrunner (ViewKit│  two-line ASCII req  │ octane_bridge  (0.0.0.0:1974)│
  │  + Motif + XImage) ├─────────────────────►│   │ JSON job (out=/tmp/*.png) │
  │                    │◄─────────────────────┤   ▼                          │
  │  XmDrawingArea     │  magic|status|w|h|RGB│ serve (worker, 127.0.0.1:8765)│
  └────────────────────┘                      └────────────────────────────┘
```

The bridge does everything awkward for a retro client — JSON, PNG decode — so the
Octane side is pure Xlib: it sends a prompt plus three integers and receives raw
RGB it drops into an `XImage`. The CUDA worker (`serve.cu`) is untouched.

## 1. On the Spark

Build and start the worker (if it isn't already resident) and the bridge:

```sh
# worker: loads the models once, serves on loopback :8765 (already how the web UI runs)
build/serve --port 8765 &

# bridge: LAN-facing, forwards to the worker. Defaults to the fast klein-4B model.
build/octane_bridge --port 1974 --model flux2-klein-4B
```

Bridge options:

| flag | default | meaning |
|------|---------|---------|
| `--port` | `1974` | LAN port the Octane connects to |
| `--worker-port` | `8765` | loopback worker port |
| `--worker-host` | `127.0.0.1` | worker host |
| `--model` | *(none = stock 9B)* | model name under `~/models`, or an absolute root, e.g. `flux2-klein-4B` |
| `--precision` | `fp8` | `fp8` or `nvfp4` |

`octane_bridge` builds with the rest of the project (`cmake --build build --target
octane_bridge`) or standalone:

```sh
g++ -O2 -std=c++17 -o build/octane_bridge tools/octane_bridge.cpp \
    -Ithird_party/json/single_include -Ithird_party/stb
```

Smoke-test it without an SGI in the loop:

```sh
.venv/bin/python tools/octane_test_client.py --list                 # what models?
.venv/bin/python tools/octane_test_client.py "a red sports car" \
    --model flux2-klein-4B --res 512 --steps 4 --out /tmp/octane_out.png
```

## 2. On the Octane

Copy this `tools/octane/` directory over, then:

```sh
make
./roadrunner -host sparky -port 1974      # or set F2K_BRIDGE_HOST / F2K_BRIDGE_PORT
```

`-host` should resolve to the Spark on your LAN — a `/etc/hosts` entry or a plain
IP (`-host 192.168.1.42`) is simplest at a booth.

Using it:

- **Model** — a dropdown, populated at startup by asking the bridge (`LIST`) what
  checkpoints it has (klein-4B, 9B, character LoRAs, finetunes…). If the bridge is
  down at launch it shows `(default)` and the bridge picks its configured default.
- **Resolution / Steps / Seed / Batch** — blank seed = random; a batch of N uses
  seed, seed+1, … (or random each). The exact seed of every image comes back with it.
- **Generate** — the UI **stays live** while the GPU works (the socket is serviced by
  the Xt event loop via `XtAppAddInput`, not a blocking read). The white **progress
  bar** tracks real phases streamed from the worker — `loading → encoding → denoise
  step k/N → decoding` — advancing across the whole batch.
- **Mode: Generate / Remix** — in **Remix** mode the *Load init image…* button and
  *Strength %* slider light up. Load a PNG/JPEG/BMP (decoded by bundled `stb_image`);
  it's centre-cropped to square, capped at 1024, previewed in the canvas, and shipped
  with the next Generate as an img2img base (lower strength = closer to the original).
- **Batch strip + navigation** — each finished image drops into the thumbnail grid
  under the controls; click a thumbnail (or **‹‹ Prev / Next ››**) to show it big in
  the canvas. The shown one is outlined.
- **Zoom / pan** — the canvas fits the image to the window by default (so a 1024 in a
  small window is never clipped). **Zoom− / Fit / 1:1 / Zoom+** buttons scale it, and
  you **drag with the left button to pan** when zoomed in (SGI mice have no wheel).
- **File → Save Image + Params…** — saves the *currently shown* image: `<name>.png`
  (real PNG, via bundled `stb_image_write`) plus a `<name>.txt` sidecar with prompt,
  model, res, steps, that image's exact seed, and size — enough to reproduce it.

### Build notes (things you may need to nudge on your IRIX)

The load-bearing code (BSD sockets, `XImage` packing via visual masks, `ntohl`
byte-order handling) is plain Xlib and should be solid. The ViewKit surface is
small and conventional, but if your ViewKit/Motif vintage differs:

- **`XmStringCreateLocalized`** / **`XmFONTLIST_DEFAULT_TAG`** are Motif 2.x. On
  older Motif use `XmStringCreateSimple` / `XmStringCreateLtoR` and
  `XmSTRING_DEFAULT_CHARSET`.
- **`stb_image_write.h`** (PNG save) and **`stb_image.h`** (decode a loaded init
  image) are bundled and included with their implementations into `roadrunner.cxx`.
  If the MIPSpro C++ front end fights either, move that `#define ..._IMPLEMENTATION`
  + include into a `.c` file, compile it with `cc`, and link it in.
- **`XmCreateScrolledText`** returns the `Text` widget; its geometry parent is
  `XtParent(_prompt)` (the ScrolledWindow) — already handled.
- The menu bar is wired with `XmMainWindowSetAreas` after ViewKit's `addView`; if
  your ViewKit reparents the work area, that's the line to check.
- The Xlib `Visual` struct's `class` member is spelled **`c_class`** under C++
  (Xlib guards the keyword). Already used.
- Only **TrueColor / DirectColor** visuals are handled (any modern Octane demo
  will be one). PseudoColor would need colormap allocation.
- Link line lives in the `Makefile`: `-lvk -lXm -lXt -lXext -lX11 -lgen -lm`.

## Wire protocol (bridge ↔ Octane)

Two request types, each a plain ASCII line:

```
LIST\n                                          -> model names, one per line, then EOF
GEN <res> <steps> <seed> <count> <model>\n      <model> is a name, or "-" = default
<prompt>\n                                      (second line, GEN only)

REMIX <res> <steps> <seed> <count> <strengthx100> <imgW> <imgH> <model>\n
<prompt>\n
<imgW*imgH*3 raw RGB bytes>                      client pre-crops to square
```

`seed < 0` → the bridge randomises. The `GEN` response is a **stream of tagged
messages** (a batch → many). Each starts with the 4-byte magic `F2K1` then an
`int32` type; every `int32` is **network byte order** (the Octane is big-endian MIPS):

```
PROGRESS (1): u32 imgIndex, imgTotal, permille(0..1000), len, phase[len]
IMAGE    (2): u32 imgIndex, w, h, seed, then w*h*3 RGB bytes (top row first)
DONE     (3): u32 count                       (batch complete)
ERROR    (4): u32 len, message[len]           (aborts the batch)
```

A batch is `PROGRESS* (IMAGE PROGRESS*)* DONE` — progress and images interleaved,
one `IMAGE` per requested count, terminated by `DONE` (or `ERROR`). Progress phases
are `starting / loading / encoding / denoise / decoding`; the worker emits these
cheaply (no preview render) only when the bridge asks (`preview:false`), so the web
UI is unaffected.
