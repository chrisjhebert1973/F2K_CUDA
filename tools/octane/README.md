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
- **Resolution / Steps / Seed** — blank seed = random; the seed the bridge actually
  used comes back with the image and shows in the status line.
- **Generate** — the request goes out and the UI **stays live** while the GPU works
  (the socket is serviced by the Xt event loop via `XtAppAddInput`, not a blocking
  read); the button greys out until the picture arrives.
- **File → Save Image + Params…** — writes `<name>.png` (real PNG, via bundled
  `stb_image_write`) plus a `<name>.txt` sidecar with prompt, model, res, steps, the
  exact seed, and size — enough to reproduce the image later.

### Build notes (things you may need to nudge on your IRIX)

The load-bearing code (BSD sockets, `XImage` packing via visual masks, `ntohl`
byte-order handling) is plain Xlib and should be solid. The ViewKit surface is
small and conventional, but if your ViewKit/Motif vintage differs:

- **`XmStringCreateLocalized`** / **`XmFONTLIST_DEFAULT_TAG`** are Motif 2.x. On
  older Motif use `XmStringCreateSimple` / `XmStringCreateLtoR` and
  `XmSTRING_DEFAULT_CHARSET`.
- **`stb_image_write.h`** is bundled and included with its implementation into
  `roadrunner.cxx`. If the MIPSpro C++ front end fights it, move the
  `#define STB_IMAGE_WRITE_IMPLEMENTATION` + include into a `stbiw.c`, compile that
  with `cc`, and link it in (drop the `#define` in the `.cxx`).
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
LIST\n                                  -> newline-separated model names, then EOF
GEN <res> <steps> <seed> <model>\n      <model> is a name, or "-" for the default
<prompt>\n                              (second line, GEN only)
```

`seed < 0` → the bridge randomises. Generation response — binary, every `int32` in
**network byte order** (the Octane is big-endian MIPS):

```
'F' '2' 'K' '1'   magic (4 bytes)
int32 status      0 = ok, non-zero = error
  ok:  int32 width, int32 height, int32 seed, width*height*3 bytes RGB (top row first)
  err: int32 msglen, msglen bytes of ASCII
```
