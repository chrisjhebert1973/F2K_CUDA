#!/usr/bin/env python3
"""F2K_CUDA web UI (v2) — a personal Flask front-end for the persistent worker,
meant to be reached over Tailscale (e.g. from an iPad). Single-user session
login; jobs go to ./build/serve (models resident, ~8s/image), with a one-shot
./build/generate fallback if the worker is down.

Run:
  F2K_WEB_PASSWORD=yourpw .venv/bin/python tools/webui/app.py
Then on any tailnet device:  http://spark-f42d:5000

Env:
  F2K_WEB_USER      login username   (default: chris)
  F2K_WEB_PASSWORD  login password   (default: rocket)   <-- change me
  F2K_SECRET        flask secret key (default: random per start)
  F2K_WEB_PORT      port             (default: 5000)
  F2K_WORKER_PORT   worker tcp port  (default: 8765)

Run metadata (one JSON per run, in runs/):
  {id, prompt, res, precision, steps, mode, when, elapsed, count, ok, error,
   images:[{seed, file, ok, nans, timing:[...]}, ...]}
Older single-image runs (no `images` key, png named <id>.png) are normalised on
read by _normalise().
"""
import os, re, glob, json, time, hmac, secrets, subprocess, threading, socket, base64
from functools import wraps
from flask import (Flask, request, session, redirect, url_for, abort,
                   render_template_string, send_from_directory, send_file, flash, Response)
from werkzeug.security import safe_join

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERATE = os.path.join(ROOT, "build", "generate")
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(RUNS, exist_ok=True)
THUMBS = os.path.join(RUNS, ".thumbs")     # cached low-res gallery previews
os.makedirs(THUMBS, exist_ok=True)
THUMB_MAX = 384                            # longest-edge px for grid thumbnails
GALLERY_PER_PAGE = 64                      # default thumbnails per gallery page
PER_PAGE_CHOICES = [32, 64, 128]

USER = os.environ.get("F2K_WEB_USER", "chris")
PASSWORD = os.environ.get("F2K_WEB_PASSWORD", "rocket")
WORKER_HOST = "127.0.0.1"
WORKER_PORT = int(os.environ.get("F2K_WORKER_PORT", "8765"))
RES_CHOICES = ["256", "512", "768", "1024"]
PREC_CHOICES = ["fp8", "nvfp4"]
COUNT_CHOICES = [1, 2, 4, 8]
MAX_COUNT = 8
MODELS_ROOT = os.path.expanduser("~/models")
STOCK_MODEL = "flux2-klein-9B"
DEFAULT_MODEL = os.environ.get("F2K_DEFAULT_MODEL", "flux2-klein-4B")
_PREC_SUBDIR = {"fp8": "transformer_mxfp8", "nvfp4": "transformer_f2k"}

def list_models():
    """Checkpoint roots the worker can serve: any ~/models/<name> holding a
    transformer_* dir with .f2k1 shards (see tools/bfl_to_diffusers.py for how
    finetunes get there). Stock klein-9B first."""
    out = []
    try:
        for name in sorted(os.listdir(MODELS_ROOT)):
            if any(glob.glob(os.path.join(MODELS_ROOT, name, sub, "*.f2k1"))
                   for sub in _PREC_SUBDIR.values()):
                out.append(name)
    except OSError:
        pass
    if STOCK_MODEL in out:
        out.remove(STOCK_MODEL); out.insert(0, STOCK_MODEL)
    return out

def _model_fields(model, precision):
    """Worker payload fields {model, transformer} for a selected checkpoint.
    Full model roots (own qwen3_f2k encoder, e.g. klein-4B) → model=<root>;
    transformer-only dirs (klein-9B finetunes like truev2) → overlay on stock.
    None if the requested quant doesn't exist for this checkpoint."""
    if not model or model == STOCK_MODEL:
        return {"model": "", "transformer": ""}
    root = os.path.join(MODELS_ROOT, model)
    tf = os.path.join(root, _PREC_SUBDIR.get(precision, "transformer_mxfp8"))
    if not glob.glob(os.path.join(tf, "*.f2k1")):
        return None
    if os.path.isdir(os.path.join(root, "qwen3_f2k")):
        return {"model": root, "transformer": ""}
    return {"model": "", "transformer": tf}

def _load_triggers():
    """Map @mention -> trained trigger phrase, from <model_root>/trigger.json.
    Each character checkpoint (train_character.sh) drops a trigger.json like
    {"mention": "rocket", "phrase": "r0cket, a black and white Akita husky dog"}.
    The phrase carries both the rare trigger token AND the class anchor, so a
    bare "@rocket ..." prompt can never collide with a base-model concept (the
    spaceship problem)."""
    out = {}
    try:
        for name in os.listdir(MODELS_ROOT):
            tj = os.path.join(MODELS_ROOT, name, "trigger.json")
            if not os.path.isfile(tj):
                continue
            try:
                d = json.load(open(tj))
                m = (d.get("mention") or "").strip().lower()
                if m and d.get("phrase"):
                    out[m] = d["phrase"]
            except Exception:
                pass
    except OSError:
        pass
    return out

def expand_triggers(prompt):
    """Rewrite @mention -> trigger phrase before the prompt reaches the encoder.
    Unknown @mentions are left untouched."""
    triggers = _load_triggers()
    if not triggers or "@" not in prompt:
        return prompt
    return re.sub(r"@([A-Za-z0-9_]+)",
                  lambda mo: triggers.get(mo.group(1).lower(), mo.group(0)),
                  prompt)

app = Flask(__name__)
app.secret_key = os.environ.get("F2K_SECRET", secrets.token_hex(16))
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024   # camera photos can be big
_gpu_lock = threading.Lock()   # serialize GPU jobs (one batch at a time)

# ----------------------------------------------------------------- auth
def login_required(f):
    @wraps(f)
    def g(*a, **k):
        if not session.get("auth"): return redirect(url_for("login"))
        return f(*a, **k)
    return g

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("user", ""); p = request.form.get("password", "")
        if hmac.compare_digest(u, USER) and hmac.compare_digest(p, PASSWORD):
            session["auth"] = True
            return redirect(url_for("index"))
        flash("Wrong username or password.")
    return render_template_string(LOGIN)

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

# ----------------------------------------------------------------- run metadata
def _normalise(m):
    """Bring any run dict up to the v2 schema (in place) and return it.
    Old runs stored a single top-level `seed` and an image at <id>.png with no
    `images` list; synthesise one so the rest of the UI is batch-uniform."""
    if "images" not in m:
        m["images"] = [dict(seed=m.get("seed", 0), file=m["id"] + ".png",
                            ok=m.get("ok", False), nans=m.get("nans", 0),
                            timing=m.get("timing", []))]
    m.setdefault("count", len(m["images"]))
    m.setdefault("ok", any(im.get("ok") for im in m["images"]))
    return m

def _meta_path(rid):
    return os.path.join(RUNS, rid + ".json")

def load_run(rid):
    p = _meta_path(rid)
    if not os.path.exists(p): return None
    try: return _normalise(json.load(open(p)))
    except Exception: return None

def save_run(m):
    json.dump(m, open(_meta_path(m["id"]), "w"), indent=2)

def list_runs():
    out = []
    for fn in sorted(os.listdir(RUNS), reverse=True):
        if fn.endswith(".json"):
            try: out.append(_normalise(json.load(open(os.path.join(RUNS, fn)))))
            except Exception: pass
    return out

def list_images(runs=None):
    """Flatten runs into per-image cards (newest first), only successful ones."""
    cards = []
    for m in (runs if runs is not None else list_runs()):
        for idx, im in enumerate(m["images"]):
            if im.get("ok"):
                cards.append(dict(rid=m["id"], idx=idx, file=im["file"],
                                  prompt=m.get("prompt", ""), seed=im.get("seed", 0)))
    return cards

def gallery_view(page=1, per_page=GALLERY_PER_PAGE):
    """For the gallery: a single page of successful images, grouped by batch
    (run) for display, plus a flat list the lightbox pages through (prev/next
    flows across the current page). Each thumbnail carries its index into
    `flat`. Returns (batches, flat, pg) where pg holds pagination metadata."""
    # Flatten every successful image newest-first, then slice to the page.
    all_imgs = []
    for m in list_runs():
        for im in m["images"]:
            if im.get("ok"):
                all_imgs.append((m, im))
    total = len(all_imgs)
    pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    chunk = all_imgs[start:start + per_page]

    batches, flat, cur = [], [], None
    for m, im in chunk:
        gi = len(flat)
        flat.append(dict(file=im["file"], prompt=m.get("prompt", ""),
                         seed=im.get("seed", 0), rid=m["id"]))
        if cur is None or cur["rid"] != m["id"]:
            cur = dict(rid=m["id"], prompt=m.get("prompt", ""),
                       when=m.get("when", ""), kind=m.get("kind", ""), images=[])
            batches.append(cur)
        cur["images"].append(dict(gi=gi, file=im["file"], seed=im.get("seed", 0)))
    for b in batches:
        b["count"] = len(b["images"])
    pg = dict(page=page, pages=pages, total=total, per_page=per_page,
              start=start + 1 if total else 0, end=start + len(chunk))
    return batches, flat, pg

# ----------------------------------------------------------------- worker
def _worker_call(payload, timeout=600):
    """Send one job to the persistent worker; raises OSError if it's not up."""
    s = socket.create_connection((WORKER_HOST, WORKER_PORT), timeout=10)
    s.settimeout(timeout)
    try:
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk: break
            buf += chunk
    finally:
        s.close()
    return json.loads(buf)

def _worker_stream(payload, timeout=600):
    """Generator: send a streaming job to the worker and yield each JSON message
    (progress events, then the final result). Raises OSError if the worker is down."""
    s = socket.create_connection((WORKER_HOST, WORKER_PORT), timeout=10)
    s.settimeout(timeout)
    try:
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk: break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip(): continue
                j = json.loads(line)
                yield j
                if j.get("event") != "progress":
                    return
    finally:
        s.close()

def _subprocess_call(out_png, prompt, res, precision, steps, seed, fields=None):
    """Fallback: one-shot generate binary (slow — reloads the model)."""
    args = [GENERATE, "--prompt", prompt, "--res", str(res), "--precision", precision,
            "--steps", str(steps), "--seed", str(seed), "--out", out_png]
    if fields and fields.get("model"):
        args += ["--model", fields["model"]]
    if fields and fields.get("transformer"):
        args += ["--transformer", fields["transformer"]]
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=600)
    log = (proc.stdout or "") + (proc.stderr or "")
    timing = [ln.strip() for ln in log.splitlines()
              if re.search(r"(ready in|Text encode|Denoise loop|VAE decode|Wrote)", ln)]
    return dict(ok=(proc.returncode == 0 and os.path.exists(out_png)), timing=timing,
                error=(log[-1500:] if proc.returncode != 0 else ""))

def run_batch(prompt, res, precision, steps, seeds, init_image=None, strength=None, kind=None,
              model="", extra=None, mask_image=None):
    """Generate len(seeds) images for one prompt, serialised on the GPU lock.
    If init_image (abs path) is given, runs img2img at the given strength — only
    the persistent worker supports this; the subprocess fallback is txt2img.
    `kind` tags the run (e.g. 'upscale 2048px') for the result banner.
    `model` selects an alternate checkpoint from list_models() ('' = stock)."""
    prompt = expand_triggers(prompt)
    rid = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
    images, mode, err = [], "worker", ""
    fields = _model_fields(model, precision)
    if fields is None:   # checkpoint exists but not in this quant
        fields = {"model": "", "transformer": ""}
        model = STOCK_MODEL
    t0 = time.time()
    with _gpu_lock:
        for i, seed in enumerate(seeds):
            fn = f"{rid}_{i}.png"
            out_png = os.path.join(RUNS, fn)
            payload = dict(prompt=prompt, res=res, precision=precision,
                           steps=steps, seed=seed, out=out_png, **fields)
            if init_image:
                payload["init_image"] = init_image
                payload["strength"] = strength
            if mask_image:
                payload["mask_image"] = mask_image
            if extra:
                payload.update(extra)
            try:
                r = _worker_call(payload)
            except OSError:
                mode = "subprocess (worker down)"
                r = _subprocess_call(out_png, prompt, res, precision, steps, seed, fields)
            if not r.get("ok") and not err:
                err = r.get("error", "")
            images.append(dict(seed=seed, file=fn, ok=r.get("ok", False),
                               nans=r.get("nans", 0), timing=r.get("timing", [])))
    meta = dict(id=rid, prompt=prompt, res=res, precision=precision, steps=steps,
                count=len(seeds), ok=any(im["ok"] for im in images),
                elapsed=round(time.time() - t0, 1), error=err, mode=mode,
                when=time.strftime("%Y-%m-%d %H:%M"), images=images,
                model=model or STOCK_MODEL)
    if init_image:
        meta["init_from"] = os.path.basename(init_image)
        meta["strength"] = strength
    if kind:
        meta["kind"] = kind
    if extra:
        meta.update({k: extra[k] for k in ("cfg", "negative", "var_strength", "seed_var")
                     if k in extra})
    save_run(meta)
    return meta

def _parse_extra(form):
    """Advanced worker fields — negative prompt, CFG scale, variation seed/strength.
    Only non-default keys are returned, so a plain request stays byte-identical."""
    out = {}
    try: cfg = float(form.get("cfg", "1") or 1)
    except ValueError: cfg = 1.0
    cfg = max(0.0, min(10.0, cfg))
    if abs(cfg - 1.0) > 1e-3:
        out["cfg"] = cfg
        out["negative"] = (form.get("negative", "") or "").strip()[:800]
    try: vstr = float(form.get("var_strength", "0") or 0)
    except ValueError: vstr = 0.0
    vstr = max(0.0, min(1.0, vstr))
    if vstr > 0:
        out["var_strength"] = round(vstr, 2)
        sv = (form.get("seed_var", "") or "").strip()
        try: out["seed_var"] = (int(sv) & 0xFFFFFFFF) if sv else secrets.randbits(32)
        except ValueError: out["seed_var"] = secrets.randbits(32)
    return out

# ----------------------------------------------------------------- form parsing
def _parse_common(form):
    res = form.get("res", "1024"); res = res if res in RES_CHOICES else "1024"
    precision = form.get("precision", "fp8")
    precision = precision if precision in PREC_CHOICES else "fp8"
    try: steps = max(1, min(30, int(form.get("steps", "4"))))
    except ValueError: steps = 4
    try: count = int(form.get("count", "1"))
    except ValueError: count = 1
    count = count if count in COUNT_CHOICES else 1
    seed_in = (form.get("seed", "") or "").strip()
    if seed_in:
        try: base = int(seed_in) & 0xFFFFFFFF
        except ValueError: base = secrets.randbits(32)
        seeds = [(base + i) & 0xFFFFFFFF for i in range(count)]   # reproducible batch
    else:
        seeds = [secrets.randbits(32) for _ in range(count)]
    model = form.get("model", "")
    if model and model not in list_models():
        model = ""
    return int(res), precision, steps, seeds, model

# ----------------------------------------------------------------- routes
@app.route("/")
@login_required
def index():
    models = list_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else STOCK_MODEL
    return render_template_string(INDEX, res_choices=RES_CHOICES, prec_choices=PREC_CHOICES,
                                  count_choices=COUNT_CHOICES, model_choices=models,
                                  default_model=default_model, cards=list_images()[:12])

@app.route("/generate", methods=["POST"])
@login_required
def generate():
    prompt = (request.form.get("prompt", "") or "").strip()[:800]
    if not prompt:
        flash("Enter a prompt."); return redirect(url_for("index"))
    res, precision, steps, seeds, model = _parse_common(request.form)
    meta = run_batch(prompt, res, precision, steps, seeds, model=model)
    return redirect(url_for("result", rid=meta["id"]))

@app.route("/generate_stream", methods=["POST"])
@login_required
def generate_stream():
    """Streaming generate (NDJSON): relays the worker's live preview events to the
    browser, saves the run, then emits {event:complete,rid}. Handles plain txt2img
    and remix-from-an-existing-output (src_rid/src_idx + strength)."""
    prompt = (request.form.get("prompt", "") or "").strip()[:800]
    res, precision, steps, seeds, model = _parse_common(request.form)
    # optional remix init from an existing output
    init_image = None; strength = None; kind = None
    src = load_run(request.form.get("src_rid", "")) if request.form.get("src_rid") else None
    if src and not model:
        model = src.get("model", "")   # remix/inpaint inherit the source checkpoint
    if src:
        try: si = int(request.form.get("src_idx", "0"))
        except ValueError: si = 0
        if 0 <= si < len(src["images"]):
            cand = os.path.join(RUNS, os.path.basename(src["images"][si]["file"]))
            if os.path.exists(cand):
                init_image = cand
                try: strength = max(0.05, min(1.0, float(request.form.get("strength", "0.6"))))
                except ValueError: strength = 0.6
                strength = round(strength, 2)
                if not prompt: prompt = src.get("prompt", "")
    if not prompt:
        return Response('{"event":"error","msg":"empty prompt"}\n', mimetype="application/x-ndjson")
    prompt = expand_triggers(prompt)
    # optional inpaint mask (painted in the browser), as a data URL
    mask_bytes = None
    md = request.form.get("mask_data", "")
    if init_image and md.startswith("data:image"):
        try: mask_bytes = base64.b64decode(md.split(",", 1)[1])
        except Exception: mask_bytes = None

    fields = _model_fields(model, precision)
    if fields is None:   # checkpoint exists but not in this quant
        fields = {"model": "", "transformer": ""}
        model = STOCK_MODEL
    extra = _parse_extra(request.form)   # read in request context, before streaming

    def gen():
        rid = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
        images, mode, err = [], "worker", ""
        mask_path = None
        if mask_bytes and init_image:
            mask_path = os.path.join(RUNS, f"mask_{rid}.png")
            with open(mask_path, "wb") as f: f.write(mask_bytes)
        t0 = time.time()
        with _gpu_lock:
            for i, seed in enumerate(seeds):
                fn = f"{rid}_{i}.png"
                out_png = os.path.join(RUNS, fn)
                payload = dict(prompt=prompt, res=res, precision=precision,
                               steps=steps, seed=seed, out=out_png, stream=True,
                               **fields)
                if init_image:
                    payload["init_image"] = init_image; payload["strength"] = strength
                if mask_path:
                    payload["mask_image"] = mask_path
                if extra:
                    payload.update(extra)
                final = None
                try:
                    for j in _worker_stream(payload):
                        if j.get("event") == "progress":
                            j["img"] = i; j["nimg"] = len(seeds)
                            yield json.dumps(j) + "\n"
                        else:
                            final = j
                except OSError:
                    mode = "subprocess (worker down)"
                    final = _subprocess_call(out_png, prompt, res, precision, steps, seed, fields)
                final = final or {"ok": False, "error": "no response"}
                if not final.get("ok") and not err: err = final.get("error", "")
                images.append(dict(seed=seed, file=fn, ok=final.get("ok", False),
                                   nans=final.get("nans", 0), timing=final.get("timing", [])))
                yield json.dumps({"event": "image_done", "img": i, "file": fn,
                                  "ok": final.get("ok", False), "nimg": len(seeds)}) + "\n"
        meta = dict(id=rid, prompt=prompt, res=res, precision=precision, steps=steps,
                    count=len(seeds), ok=any(im["ok"] for im in images),
                    elapsed=round(time.time() - t0, 1), error=err, mode=mode,
                    when=time.strftime("%Y-%m-%d %H:%M"), images=images,
                    model=model or STOCK_MODEL)
        if extra:
            meta.update({k: extra[k] for k in ("cfg", "negative", "var_strength", "seed_var")
                         if k in extra})
        if init_image:
            meta["init_from"] = os.path.basename(init_image); meta["strength"] = strength
        if mask_path: meta["kind"] = "inpaint"
        elif kind: meta["kind"] = kind
        save_run(meta)
        yield json.dumps({"event": "complete", "rid": rid}) + "\n"

    return Response(gen(), mimetype="application/x-ndjson",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

@app.route("/result/<rid>")
@login_required
def result(rid):
    m = load_run(rid)
    if not m: abort(404)
    return render_template_string(RESULT, m=m)

@app.route("/remix/<rid>/<int:idx>")
@login_required
def remix_form(rid, idx):
    m = load_run(rid)
    if not m or idx < 0 or idx >= len(m["images"]): abort(404)
    im = m["images"][idx]
    if not im.get("ok"): abort(404)
    return render_template_string(REMIX, m=m, im=im, idx=idx, res_choices=RES_CHOICES,
                                  prec_choices=PREC_CHOICES, count_choices=COUNT_CHOICES)

@app.route("/inpaint/<rid>/<int:idx>")
@login_required
def inpaint_form(rid, idx):
    m = load_run(rid)
    if not m or idx < 0 or idx >= len(m["images"]): abort(404)
    im = m["images"][idx]
    if not im.get("ok"): abort(404)
    return render_template_string(INPAINT, m=m, im=im, idx=idx, res_choices=RES_CHOICES,
                                  prec_choices=PREC_CHOICES, count_choices=COUNT_CHOICES)

@app.route("/remix", methods=["POST"])
@login_required
def remix():
    src = load_run(request.form.get("src_rid", ""))
    try: src_idx = int(request.form.get("src_idx", "0"))
    except ValueError: src_idx = 0
    if not src or src_idx < 0 or src_idx >= len(src["images"]): abort(404)
    init_path = os.path.join(RUNS, os.path.basename(src["images"][src_idx]["file"]))
    if not os.path.exists(init_path): abort(404)
    prompt = (request.form.get("prompt", "") or "").strip()[:800] or src.get("prompt", "")
    res, precision, steps, seeds, model = _parse_common(request.form)
    if not model:
        model = src.get("model", "")   # remix inherits the source checkpoint
    try: strength = float(request.form.get("strength", "0.6"))
    except ValueError: strength = 0.6
    strength = max(0.05, min(1.0, strength))
    meta = run_batch(prompt, res, precision, steps, seeds,
                     init_image=init_path, strength=round(strength, 2), model=model)
    return redirect(url_for("result", rid=meta["id"]))

def _save_upload_square(file_storage, maxdim=1024):
    """Read an uploaded image, center-crop to square, downscale to <=maxdim, and
    save as PNG in runs/. Returns the abs path (or None on failure). The model
    only generates square images, so center-cropping beats squashing."""
    from PIL import Image, ImageOps
    try:
        img = ImageOps.exif_transpose(Image.open(file_storage.stream)).convert("RGB")
    except Exception:
        return None
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if s > maxdim:
        img = img.resize((maxdim, maxdim), Image.BICUBIC)
    name = f"upload_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}.png"
    path = os.path.join(RUNS, name)
    img.save(path, "PNG")
    return path

@app.route("/upload_remix", methods=["POST"])
@login_required
def upload_remix():
    f = request.files.get("photo")
    if not f or not f.filename:
        flash("Choose a photo first."); return redirect(url_for("index"))
    prompt = (request.form.get("prompt", "") or "").strip()[:800]
    if not prompt:
        flash("Add a prompt describing how to remix the photo."); return redirect(url_for("index"))
    init_path = _save_upload_square(f)
    if not init_path:
        flash("Could not read that image (HEIC isn't supported — use JPEG/PNG).")
        return redirect(url_for("index"))
    res, precision, steps, seeds, model = _parse_common(request.form)
    try: strength = float(request.form.get("strength", "0.55"))
    except ValueError: strength = 0.55
    strength = max(0.05, min(1.0, strength))
    meta = run_batch(prompt, res, precision, steps, seeds,
                     init_image=init_path, strength=round(strength, 2), model=model)
    return redirect(url_for("result", rid=meta["id"]))

UPSCALE_MAX = 2048        # true 2K (Linear gridDim.y fix unblocked 65536 tokens)
UPSCALE_STRENGTH = 0.35   # low: add detail at higher res, keep the content

@app.route("/upscale/<rid>/<int:idx>", methods=["POST"])
@login_required
def upscale(rid, idx):
    m = load_run(rid)
    if not m or idx < 0 or idx >= len(m["images"]): abort(404)
    im = m["images"][idx]
    if not im.get("ok"): abort(404)
    init_path = os.path.join(RUNS, os.path.basename(im["file"]))
    if not os.path.exists(init_path): abort(404)
    src_res = int(m.get("res", 1024))
    target = min(src_res * 2, UPSCALE_MAX)
    if target <= src_res:
        flash("Already at the maximum size."); return redirect(url_for("result", rid=rid))
    steps = max(int(m.get("steps", 8)), 12)        # a few more steps help at 2K
    seeds = [im.get("seed", secrets.randbits(32))]  # reuse seed for consistency
    meta = run_batch(m.get("prompt", ""), target, m.get("precision", "fp8"), steps, seeds,
                     init_image=init_path, strength=UPSCALE_STRENGTH, kind=f"upscale {target}px",
                     model=m.get("model", ""))
    return redirect(url_for("result", rid=meta["id"]))

OUTPAINT_ZOOM = 2.0       # zoom-out factor: the source shrinks to 1/zoom of the frame
OUTPAINT_STRENGTH = 0.9   # fill the new border fairly freely

def _make_outpaint(src_path, out_res, zoom=OUTPAINT_ZOOM):
    """Zoom-out outpaint: put a shrunken copy of the source in the centre of an
    out_res square, and return (canvas_path, mask_path) for the inpaint worker.
    White mask = the new border to fill; a blurred zoom seeds it with plausible
    colour and the seam is feathered so it blends."""
    from PIL import Image, ImageDraw, ImageFilter
    inner = max(64, int(out_res / zoom))
    off = (out_res - inner) // 2
    src = Image.open(src_path).convert("RGB").resize((inner, inner), Image.LANCZOS)
    canvas = (Image.open(src_path).convert("RGB")
              .resize((out_res, out_res), Image.LANCZOS)
              .filter(ImageFilter.GaussianBlur(30)))       # blurred fill for the border
    canvas.paste(src, (off, off))
    mask = Image.new("L", (out_res, out_res), 255)         # regenerate everywhere...
    ImageDraw.Draw(mask).rectangle([off, off, off + inner - 1, off + inner - 1], fill=0)  # ...keep centre
    mask = mask.filter(ImageFilter.GaussianBlur(max(5, out_res // 128)))   # feather the seam
    tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
    cpath = os.path.join(RUNS, f"outpaint_{tag}.png"); canvas.save(cpath, "PNG")
    mpath = os.path.join(RUNS, f"outmask_{tag}.png");  mask.save(mpath, "PNG")
    return cpath, mpath

@app.route("/outpaint/<rid>/<int:idx>", methods=["POST"])
@login_required
def outpaint(rid, idx):
    m = load_run(rid)
    if not m or idx < 0 or idx >= len(m["images"]): abort(404)
    im = m["images"][idx]
    if not im.get("ok"): abort(404)
    src_path = os.path.join(RUNS, os.path.basename(im["file"]))
    if not os.path.exists(src_path): abort(404)
    res = int(m.get("res", 1024))
    canvas_path, mask_path = _make_outpaint(src_path, res)
    prompt = (request.form.get("prompt", "") or "").strip()[:800] or m.get("prompt", "")
    steps = max(int(m.get("steps", 8)), 12)
    meta = run_batch(prompt, res, m.get("precision", "fp8"), steps, [secrets.randbits(32)],
                     init_image=canvas_path, mask_image=mask_path,
                     strength=OUTPAINT_STRENGTH, kind="outpaint", model=m.get("model", ""))
    return redirect(url_for("result", rid=meta["id"]))

@app.route("/delete/<rid>/<int:idx>", methods=["POST"])
@login_required
def delete(rid, idx):
    m = load_run(rid)
    if not m or idx < 0 or idx >= len(m["images"]): abort(404)
    im = m["images"][idx]
    base = os.path.basename(im["file"])
    try: os.remove(os.path.join(RUNS, base))
    except OSError: pass
    try: os.remove(os.path.join(THUMBS, base.rsplit(".", 1)[0] + ".webp"))
    except OSError: pass
    m["images"].pop(idx)
    if m["images"]:
        m["count"] = len(m["images"]); m["ok"] = any(x.get("ok") for x in m["images"])
        save_run(m)
        flash("Deleted one image.")
        return redirect(url_for("result", rid=rid))
    # last image gone — drop the whole run, then return to where it sat in the
    # gallery: anchor on the neighbouring batch that now occupies that spot
    # (next-older, else previous), so the page scrolls back to that position.
    order = [r["id"] for r in list_runs()]      # newest-first, still includes rid
    neighbor = None
    if rid in order:
        i = order.index(rid)
        neighbor = order[i + 1] if i + 1 < len(order) else (order[i - 1] if i else None)
    try: os.remove(_meta_path(rid))
    except OSError: pass
    flash("Deleted.")
    dest = request.form.get("back") or url_for("gallery")
    if neighbor and "gallery" in dest and "#" not in dest:
        dest += "#b-" + neighbor
    return redirect(dest)

@app.route("/gallery")
@login_required
def gallery():
    # Query params win; otherwise fall back to the last-viewed state in the
    # session so a redirect to /gallery (e.g. after delete) lands where you were.
    per = request.args.get("per", type=int)
    if per not in PER_PAGE_CHOICES:
        per = session.get("gallery_per", GALLERY_PER_PAGE)
    page = request.args.get("page", type=int)
    if not page or page < 1:
        page = session.get("gallery_page", 1)
    batches, flat, pg = gallery_view(page, per)
    session["gallery_per"] = pg["per_page"]    # remember as the default
    session["gallery_page"] = pg["page"]       # store the clamped, valid page
    return render_template_string(GALLERY, batches=batches, flat=flat, pg=pg,
                                  per_choices=PER_PAGE_CHOICES)

@app.route("/img/<path:fn>")
@login_required
def img(fn):
    return send_from_directory(RUNS, fn)

@app.route("/thumb/<path:fn>")
@login_required
def thumb(fn):
    """Low-res WebP preview for the gallery grid, generated on first request
    and cached in runs/.thumbs (regenerated if the source PNG is newer)."""
    src = safe_join(RUNS, fn)
    if not src or not os.path.isfile(src):
        abort(404)
    tp = os.path.join(THUMBS, os.path.basename(fn).rsplit(".", 1)[0] + ".webp")
    if not os.path.exists(tp) or os.path.getmtime(tp) < os.path.getmtime(src):
        try:
            from PIL import Image
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_MAX, THUMB_MAX))
                im.save(tp, "WEBP", quality=80, method=4)
        except Exception:
            return send_from_directory(RUNS, fn)   # fall back to the full image
    return send_file(tp, mimetype="image/webp")

# ----------------------------------------------------------------- templates
BASE_CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box} body{margin:0;background:#15171c;color:#e7e9ee;
 font:16px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif}
a{color:#8fb6ff;text-decoration:none} .wrap{max-width:820px;margin:0 auto;padding:18px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
h1{font-size:20px;margin:0} h1 small{color:#8a8f9c;font-weight:400;font-size:13px}
label{display:block;margin:12px 0 4px;color:#aab0bd;font-size:14px}
input,select,textarea{width:100%;padding:12px;border-radius:10px;border:1px solid #2c3038;
 background:#1d2026;color:#e7e9ee;font-size:16px}
textarea{min-height:88px;resize:vertical}
.row{display:flex;gap:10px;flex-wrap:wrap} .row>div{flex:1;min-width:110px}
button{margin-top:16px;width:100%;padding:14px;border:0;border-radius:12px;
 background:#3b6cf0;color:#fff;font-size:17px;font-weight:600;cursor:pointer}
button:active{background:#2f57c4} .flash{background:#5a2330;padding:10px;border-radius:8px;margin:8px 0}
img.gen{width:100%;border-radius:12px;display:block;background:#000}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.card{background:#1d2026;border:1px solid #2c3038;border-radius:12px;overflow:hidden;position:relative}
.card img{width:100%;display:block} .card .c{padding:8px;font-size:12px;color:#aab0bd}
.meta{background:#1d2026;border:1px solid #2c3038;border-radius:12px;padding:12px;font-size:14px}
.meta code{color:#9fe0a0} .muted{color:#8a8f9c;font-size:13px}
.tile{background:#1d2026;border:1px solid #2c3038;border-radius:12px;overflow:hidden}
.tile img{width:100%;display:block}
.tile .bar{display:flex;gap:8px;padding:8px;flex-wrap:wrap}
.tile .bar form{flex:1;margin:0} .tile .bar button{margin:0;padding:9px;font-size:13px;border-radius:9px;width:100%}
.tile .bar .btn{flex:1;display:block;text-align:center;padding:9px;font-size:13px;border-radius:9px;
 background:#3b6cf0;color:#fff;font-weight:600}
.del{background:#5a2330} .del:active{background:#46101c}
.up{background:#1f7a6a} .up:active{background:#155448}
.op{background:#b5651d} .op:active{background:#8a4d16}
.ip{background:#6c4bd6} .ip:active{background:#553aae}
.thumb{width:120px;height:120px;object-fit:cover;border-radius:10px;border:1px solid #2c3038}
output.sv{color:#9fe0a0;font-variant-numeric:tabular-nums}
.tile img{cursor:zoom-in}
#lb{display:none;position:fixed;inset:0;background:#000e;z-index:20;align-items:center;
 justify-content:center;padding:10px}
#lb img{max-width:96vw;max-height:92vh;object-fit:contain;border-radius:10px}
#lb .x{position:fixed;top:10px;right:16px;font-size:32px;line-height:1;color:#fff;opacity:.85}
#ov{display:none;position:fixed;inset:0;background:#000b;align-items:center;justify-content:center;
 z-index:9;flex-direction:column;gap:14px;text-align:center;padding:20px}
.spin{width:46px;height:46px;border:5px solid #3b6cf0;border-top-color:transparent;border-radius:50%;
 animation:s 1s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
.batch{margin:0 0 16px}
.batch .bh{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin:0 2px 6px}
.batch .bh a{font-size:14px} .batch .bh .muted{flex:0 0 auto;font-size:12px}
.gthumb{width:100%;aspect-ratio:1;object-fit:cover;border-radius:10px;border:1px solid #2c3038;
 cursor:zoom-in;display:block}
#lb .nav{position:fixed;top:50%;transform:translateY(-50%);font-size:46px;line-height:1;color:#fff;
 opacity:.55;padding:8px 14px;cursor:pointer;user-select:none;-webkit-user-select:none}
#lb .nav:active{opacity:1} #lbprev{left:4px} #lbnext{right:4px}
#lb .cap{position:fixed;left:0;right:0;bottom:0;padding:10px 14px;background:#000a;font-size:13px;
 color:#cfd3db;text-align:center} #lb .cap a{color:#8fb6ff}
#lb .cap .n{color:#8a8f9c;font-variant-numeric:tabular-nums}
.pager{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;
 margin:0 2px 16px;padding:10px 12px;background:#1d2026;border:1px solid #2c3038;border-radius:12px}
.pager .pg{display:flex;align-items:center;gap:10px}
.pager .pg a,.pager .pg b{padding:4px 8px;border-radius:8px}
.pager .pg a{background:#262a31} .pager .pg b{background:#3b6cf0;color:#fff}
.pager .dis{padding:4px 8px;color:#5a606c}
.batch.flash-batch{animation:bflash 1.6s ease-out}
@keyframes bflash{0%{box-shadow:0 0 0 3px #3b6cf0}60%{box-shadow:0 0 0 3px #3b6cf0}100%{box-shadow:0 0 0 3px #3b6cf000}}
"""
STREAM_JS = """<script>
async function streamSubmit(form){
 var ov=document.getElementById('ov'); ov.style.display='flex';
 var img=document.getElementById('ovimg'), sub=document.getElementById('ovsub');
 if(img){img.style.display='none';} if(sub){sub.textContent='starting…';}
 try{
  var resp=await fetch(form.dataset.stream,{method:'POST',body:new FormData(form)});
  if(!resp.ok||!resp.body) throw new Error('no stream (HTTP '+resp.status+')');
  var rd=resp.body.getReader(), dec=new TextDecoder(), buf='';
  while(true){
   var r=await rd.read(); if(r.done) break;
   buf+=dec.decode(r.value,{stream:true}); var nl;
   while((nl=buf.indexOf('\\n'))>=0){
    var ln=buf.slice(0,nl); buf=buf.slice(nl+1); if(!ln.trim()) continue;
    var j=JSON.parse(ln);
    if(j.event==='progress'){ if(img){img.src='data:image/jpeg;base64,'+j.img_b64; img.style.display='block';}
      if(sub){sub.textContent='step '+j.step+'/'+j.total+(j.nimg>1?' · image '+(j.img+1)+'/'+j.nimg:'');} }
    else if(j.event==='complete'){ window.location='/result/'+j.rid; return; }
    else if(j.event==='error'){ throw new Error(j.msg||'error'); }
   }
  }
  throw new Error('stream ended early');
 }catch(e){ ov.style.display='none'; alert('Generation failed: '+(e&&e.message?e.message:e)); }
}
</script>"""

LOGIN = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · login</title><style>{{css}}</style><div class=wrap>
<header><h1>F2K_CUDA</h1></header>
{% with m=get_flashed_messages() %}{% if m %}<div class=flash>{{m[0]}}</div>{% endif %}{% endwith %}
<form method=post><label>Username</label><input name=user autocapitalize=off autocorrect=off>
<label>Password</label><input name=password type=password>
<button>Sign in</button></form></div>""".replace("{{css}}", BASE_CSS)

INDEX = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · generate</title><style>{{css}}</style>
<div id=ov><img id=ovimg style="display:none;max-width:82vw;max-height:50vh;border-radius:10px;margin-bottom:10px"><div class=spin></div><div>Generating…<br><span class=muted id=ovsub>starting…</span></div></div>
<div class=wrap><header><h1>F2K_CUDA <small>image generator</small></h1>
<a href="{{url_for('gallery')}}">Gallery</a></header>
{% with m=get_flashed_messages() %}{% if m %}<div class=flash>{{m[0]}}</div>{% endif %}{% endwith %}
<form method=post action="{{url_for('generate')}}" data-stream="{{url_for('generate_stream')}}" onsubmit="streamSubmit(this);return false">
<label>Prompt</label><textarea name=prompt placeholder="My dog Rocket, a black and white Akita husky, ..." autofocus></textarea>
<div class=row>
 <div><label>Resolution</label><select name=res>{% for r in res_choices %}<option {{'selected' if r=='1024'}}>{{r}}</option>{% endfor %}</select></div>
 <div><label>Precision</label><select name=precision>{% for p in prec_choices %}<option {{'selected' if p=='fp8'}}>{{p}}</option>{% endfor %}</select></div>
 <div><label>Steps</label><input name=steps type=number value=4 min=1 max=30></div>
</div>
<div class=row>
 <div><label>Batch</label><select name=count onchange="document.getElementById('ovsub').textContent='resident worker · ~8s × '+this.value">{% for c in count_choices %}<option value="{{c}}">{{c}} image{{'s' if c>1}}</option>{% endfor %}</select></div>
 <div><label>Seed <span class=muted>(blank=random)</span></label><input name=seed type=number placeholder=random></div>
 {% if model_choices|length > 1 %}<div><label>Model</label><select name=model>{% for mc in model_choices %}<option value="{{mc}}" {{'selected' if mc==default_model}}>{{mc.replace('flux2-klein-9B-','').replace('flux2-klein-','')}}{{' (stock)' if loop.first}}</option>{% endfor %}</select></div>{% endif %}
</div>
<details style="margin-top:6px"><summary class=muted style="cursor:pointer">Advanced · negative / guidance / variation</summary>
<label>Negative prompt <span class=muted>(only used when guidance &gt; 1)</span></label>
<textarea name=negative placeholder="blurry, low quality, extra fingers, watermark ..."></textarea>
<div class=row>
 <div><label>Guidance <output class=sv id=csv>1.0</output> <span class=muted>1 = off · &gt;2 gets trippy · ~2× slower</span></label><input name=cfg type=range min=1 max=3 step=0.1 value=1 oninput="document.getElementById('csv').value=(+this.value).toFixed(1)"></div>
 <div><label>Variation <output class=sv id=vsv>0.00</output></label><input name=var_strength type=range min=0 max=1 step=0.05 value=0 oninput="document.getElementById('vsv').value=(+this.value).toFixed(2)"></div>
 <div><label>Var seed <span class=muted>(blank=random)</span></label><input name=seed_var type=number placeholder=random></div>
</div></details>
<button>Generate</button></form>

<h1 style="font-size:15px;margin:24px 0 8px;color:#aab0bd">📷 Remix a photo</h1>
<form method=post action="{{url_for('upload_remix')}}" enctype=multipart/form-data
 onsubmit="document.getElementById('ovsub').textContent='img2img from your photo';document.getElementById('ov').style.display='flex'">
<label>Photo <span class=muted>(camera or library; center-cropped to square)</span></label>
<input type=file name=photo accept="image/*" required>
<label>Prompt</label><textarea name=prompt placeholder="e.g. as an oil painting · cyberpunk at night · turn Rocket into a bronze statue"></textarea>
<label>Strength <output class=sv id=usv>0.55</output> <span class=muted>(low = closer to the photo)</span></label>
<input name=strength type=range min=0.05 max=1.0 step=0.05 value=0.55
 oninput="document.getElementById('usv').value=(+this.value).toFixed(2)">
<div class=row>
 <div><label>Resolution</label><select name=res>{% for r in res_choices %}<option {{'selected' if r=='1024'}}>{{r}}</option>{% endfor %}</select></div>
 <div><label>Precision</label><select name=precision>{% for p in prec_choices %}<option {{'selected' if p=='fp8'}}>{{p}}</option>{% endfor %}</select></div>
 <div><label>Steps</label><input name=steps type=number value=8 min=1 max=30></div>
</div>
<div class=row>
 <div><label>Batch</label><select name=count>{% for c in count_choices %}<option value="{{c}}">{{c}} image{{'s' if c>1}}</option>{% endfor %}</select></div>
 <div><label>Seed <span class=muted>(blank=random)</span></label><input name=seed type=number placeholder=random></div>
 {% if model_choices|length > 1 %}<div><label>Model</label><select name=model>{% for mc in model_choices %}<option value="{{mc}}" {{'selected' if mc==default_model}}>{{mc.replace('flux2-klein-9B-','').replace('flux2-klein-','')}}{{' (stock)' if loop.first}}</option>{% endfor %}</select></div>{% endif %}
</div><button>Remix photo</button></form>

{% if cards %}<h1 style="font-size:15px;margin:22px 0 8px;color:#aab0bd">Recent</h1>
<div class=grid>{% for c in cards %}<a class=card href="{{url_for('result',rid=c.rid)}}">
<img src="{{url_for('img',fn=c.file)}}"><div class=c>{{c.prompt[:60]}}</div></a>{% endfor %}</div>{% endif %}
</div>{{js}}""".replace("{{css}}", BASE_CSS).replace("{{js}}", STREAM_JS)

RESULT = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · result</title><style>{{css}}</style>
<div id=ov><div class=spin></div><div>Working…<br><span class=muted id=ovsub>upscaling · ~40s (up to ~80s at 2K)</span></div></div>
<div class=wrap>
<header><h1><a href="{{url_for('index')}}">← New</a></h1><a href="{{url_for('gallery')}}">Gallery</a></header>
{% with msg=get_flashed_messages() %}{% if msg %}<div class=flash>{{msg[0]}}</div>{% endif %}{% endwith %}
{% if not m.ok %}<div class=flash>Generation failed.<pre style="white-space:pre-wrap;font-size:12px">{{m.error}}</pre></div>{% endif %}
{% if m.init_from %}<div class=meta style="margin-bottom:10px;display:flex;gap:10px;align-items:center">
<img src="{{url_for('img',fn=m.init_from)}}" style="width:56px;height:56px;object-fit:cover;border-radius:8px" onerror="this.style.display='none'">
<span class=muted>{% if m.kind and m.kind.startswith('upscale') %}⬆ Upscaled from this · {{m.kind}}{% elif m.kind=='inpaint' %}🖌 Inpainted from this · strength {{m.strength}}{% else %}🎨 Remixed from this · strength {{m.strength}}{% endif %}</span></div>{% endif %}
<div class=grid>
{% for im in m.images %}<div class=tile>
 {% if im.ok %}<img src="{{url_for('img',fn=im.file)}}" onclick="showLB(this.src)">{% else %}<div class=c style="padding:24px;text-align:center">failed</div>{% endif %}
 <div class=bar>
  {% if im.ok %}<a class=btn href="{{url_for('remix_form',rid=m.id,idx=loop.index0)}}">Remix</a>
  <a class="btn ip" href="{{url_for('inpaint_form',rid=m.id,idx=loop.index0)}}">Inpaint</a>
  {% if m.res < 2048 %}<form method=post action="{{url_for('upscale',rid=m.id,idx=loop.index0)}}" onsubmit="document.getElementById('ov').style.display='flex'">
   <button class=up title="Upscale 2×">Upscale</button></form>{% endif %}
  <form method=post action="{{url_for('outpaint',rid=m.id,idx=loop.index0)}}" onsubmit="document.getElementById('ov').style.display='flex'">
   <button class=op title="Outpaint (zoom out &amp; fill the border)">Outpaint</button></form>{% endif %}
  <form method=post action="{{url_for('delete',rid=m.id,idx=loop.index0)}}" onsubmit="return confirm('Delete this image?')">
   <button class=del>Delete</button></form>
 </div>
 <div class=c>seed {{im.seed}}{% if im.nans %} · {{im.nans}} nan{% endif %}</div>
</div>{% endfor %}
</div>
<p class=muted>Tap an image to enlarge · tap &amp; hold the large view to save it.</p>
<div class=meta><b>{{m.prompt}}</b><br>
<code>res {{m.res}} · {{m.precision}} · {{m.steps}} steps · {{m.count}} image{{'s' if m.count>1}}{% if m.model and m.model != 'flux2-klein-9B' %} · {{m.model.replace('flux2-klein-9B-','')}}{% endif %}</code><br>
<span class=muted>{{m.when}} · {{m.elapsed}}s total · {{m.mode}}</span>
</div>
<div id=lb onclick="this.style.display='none'"><span class=x>&times;</span><img id=lbimg src=""></div>
<script>function showLB(s){document.getElementById('lbimg').src=s;document.getElementById('lb').style.display='flex'}</script>
</div>""".replace("{{css}}", BASE_CSS)

REMIX = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · remix</title><style>{{css}}</style>
<div id=ov><img id=ovimg style="display:none;max-width:82vw;max-height:50vh;border-radius:10px;margin-bottom:10px"><div class=spin></div><div>Remixing…<br><span class=muted id=ovsub>starting…</span></div></div>
<div class=wrap><header><h1><a href="{{url_for('result',rid=m.id)}}">← Back</a> Remix</h1>
<a href="{{url_for('gallery')}}">Gallery</a></header>
<div class=row style="align-items:center;margin-bottom:6px">
 <img class=thumb style="flex:0 0 auto" src="{{url_for('img',fn=im.file)}}">
 <div><span class=muted>Image-to-image from this output. The structure is kept; higher
 strength re-renders more of it toward your prompt.</span></div></div>
<form method=post action="{{url_for('remix')}}" data-stream="{{url_for('generate_stream')}}" onsubmit="streamSubmit(this);return false">
<input type=hidden name=src_rid value="{{m.id}}"><input type=hidden name=src_idx value="{{idx}}">
<label>Prompt</label><textarea name=prompt autofocus>{{m.prompt}}</textarea>
<label>Strength <output class=sv id=sv>0.60</output> <span class=muted>(low = closer to original)</span></label>
<input name=strength type=range min=0.05 max=1.0 step=0.05 value=0.6
 oninput="document.getElementById('sv').value=(+this.value).toFixed(2)">
<div class=row>
 <div><label>Resolution</label><select name=res>{% for r in res_choices %}<option {{'selected' if r==(m.res|string)}}>{{r}}</option>{% endfor %}</select></div>
 <div><label>Precision</label><select name=precision>{% for p in prec_choices %}<option {{'selected' if p==m.precision}}>{{p}}</option>{% endfor %}</select></div>
 <div><label>Steps</label><input name=steps type=number value={{m.steps}} min=1 max=30></div>
</div>
<div class=row>
 <div><label>Batch</label><select name=count>{% for c in count_choices %}<option value="{{c}}">{{c}} image{{'s' if c>1}}</option>{% endfor %}</select></div>
 <div><label>Seed <span class=muted>(blank=random)</span></label><input name=seed type=number placeholder=random></div>
</div><button>Remix</button></form>
</div>{{js}}""".replace("{{css}}", BASE_CSS).replace("{{js}}", STREAM_JS)

INPAINT_JS = """<script>
(function(){
 var c=document.getElementById('mk'), ctx=c.getContext('2d'); c.width=384; c.height=384;
 var painting=false, painted=false;
 function pos(e){ var r=c.getBoundingClientRect();
   return [(e.clientX-r.left)*c.width/r.width, (e.clientY-r.top)*c.height/r.height]; }
 function dot(e){ var r=c.getBoundingClientRect(), b=+document.getElementById('brush').value;
   var p=pos(e); ctx.fillStyle='rgba(255,45,85,0.55)';
   ctx.beginPath(); ctx.arc(p[0],p[1], b*c.width/r.width/2, 0, 7); ctx.fill(); painted=true; }
 c.addEventListener('pointerdown',function(e){painting=true; dot(e); e.preventDefault();});
 c.addEventListener('pointermove',function(e){ if(painting){ dot(e); e.preventDefault(); }});
 window.addEventListener('pointerup',function(){painting=false;});
 window.clearMask=function(){ ctx.clearRect(0,0,c.width,c.height); painted=false; };
 window.inpaintSubmit=function(form){
   if(!painted){ alert('Paint over the area you want to change first.'); return false; }
   var mk=document.createElement('canvas'); mk.width=c.width; mk.height=c.height;
   var mc=mk.getContext('2d'); var src=ctx.getImageData(0,0,c.width,c.height).data;
   var out=mc.createImageData(c.width,c.height), d=out.data;
   for(var i=0;i<src.length;i+=4){ var on=src[i+3]>10?255:0; d[i]=d[i+1]=d[i+2]=on; d[i+3]=255; }
   mc.putImageData(out,0,0); form.mask_data.value=mk.toDataURL('image/png');
   streamSubmit(form); return false;
 };
})();
</script>"""

INPAINT = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · inpaint</title><style>{{css}}</style>
<div id=ov><img id=ovimg style="display:none;max-width:82vw;max-height:50vh;border-radius:10px;margin-bottom:10px"><div class=spin></div><div>Inpainting…<br><span class=muted id=ovsub>starting…</span></div></div>
<div class=wrap><header><h1><a href="{{url_for('result',rid=m.id)}}">← Back</a> Inpaint</h1>
<a href="{{url_for('gallery')}}">Gallery</a></header>
<p class=muted>Paint over the area you want to change, then describe what should go there.</p>
<div style="position:relative;width:100%;max-width:384px;margin:0 auto;touch-action:none">
 <img src="{{url_for('img',fn=im.file)}}" style="width:100%;display:block;border-radius:10px;pointer-events:none;-webkit-user-select:none">
 <canvas id=mk style="position:absolute;inset:0;width:100%;height:100%;border-radius:10px;touch-action:none"></canvas>
</div>
<div class=row style="margin-top:10px;align-items:flex-end">
 <div><label>Brush <output class=sv id=bsv>40</output></label><input id=brush type=range min=10 max=140 value=40 oninput="document.getElementById('bsv').value=this.value"></div>
 <div style="flex:0 0 auto"><button type=button class=del style="margin:0;width:auto;padding:12px 18px" onclick="clearMask()">Clear</button></div>
</div>
<form method=post action="{{url_for('generate_stream')}}" data-stream="{{url_for('generate_stream')}}" onsubmit="return inpaintSubmit(this)">
<input type=hidden name=src_rid value="{{m.id}}"><input type=hidden name=src_idx value="{{idx}}"><input type=hidden name=mask_data>
<label>Prompt</label><textarea name=prompt placeholder="what should appear in the painted area" autofocus>{{m.prompt}}</textarea>
<label>Strength <output class=sv id=sv>0.85</output> <span class=muted>(higher = fill more freely)</span></label>
<input name=strength type=range min=0.5 max=1.0 step=0.05 value=0.85 oninput="document.getElementById('sv').value=(+this.value).toFixed(2)">
<div class=row>
 <div><label>Resolution</label><select name=res>{% for r in res_choices %}<option {{'selected' if r==(m.res|string)}}>{{r}}</option>{% endfor %}</select></div>
 <div><label>Precision</label><select name=precision>{% for p in prec_choices %}<option {{'selected' if p==m.precision}}>{{p}}</option>{% endfor %}</select></div>
 <div><label>Steps</label><input name=steps type=number value=12 min=1 max=30></div>
</div>
<div class=row>
 <div><label>Batch</label><select name=count>{% for c in count_choices %}<option value="{{c}}">{{c}} image{{'s' if c>1}}</option>{% endfor %}</select></div>
 <div><label>Seed <span class=muted>(blank=random)</span></label><input name=seed type=number placeholder=random></div>
</div><button>Inpaint</button></form>
</div>{{js}}{{ijs}}""".replace("{{css}}", BASE_CSS).replace("{{js}}", STREAM_JS).replace("{{ijs}}", INPAINT_JS)

GALLERY = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · gallery</title><style>{{css}}</style><div class=wrap>
<header><h1><a href="{{url_for('index')}}">← New</a> Gallery</h1><a href="{{url_for('logout')}}">Sign out</a></header>
{% with msg=get_flashed_messages() %}{% if msg %}<div class=flash>{{msg[0]}}</div>{% endif %}{% endwith %}
{% macro pager() %}{% if pg.pages > 1 or pg.total > pg.per_page %}<div class=pager>
 <span class=muted>{{pg.start}}–{{pg.end}} of {{pg.total}}</span>
 <span class=pg>
  {% if pg.page > 1 %}<a href="{{url_for('gallery',page=pg.page-1,per=pg.per_page)}}">‹ Prev</a>{% else %}<span class=dis>‹ Prev</span>{% endif %}
  <span class=muted>page {{pg.page}} / {{pg.pages}}</span>
  {% if pg.page < pg.pages %}<a href="{{url_for('gallery',page=pg.page+1,per=pg.per_page)}}">Next ›</a>{% else %}<span class=dis>Next ›</span>{% endif %}
 </span>
 <span class=pg><span class=muted>per</span>{% for n in per_choices %}{% if n==pg.per_page %}<b>{{n}}</b>{% else %}<a href="{{url_for('gallery',page=1,per=n)}}">{{n}}</a>{% endif %}{% endfor %}</span>
</div>{% endif %}{% endmacro %}
{{ pager() }}
{% for b in batches %}<div class=batch id="b-{{b.rid}}">
 <div class=bh><a href="{{url_for('result',rid=b.rid)}}">{{ (b.prompt[:70] if b.prompt else 'untitled') }}</a>
 <span class=muted>{{b.when}} · {{b.count}} img{{'s' if b.count>1}}{% if b.kind %} · {{b.kind}}{% endif %}</span></div>
 <div class=grid>{% for e in b.images %}<img class=gthumb loading=lazy src="{{url_for('thumb',fn=e.file)}}" onclick="lbOpen({{e.gi}})">{% endfor %}</div>
</div>{% else %}<p class=muted>No images yet — <a href="{{url_for('index')}}">generate some</a>.</p>{% endfor %}
{{ pager() }}
</div>
<div id=lb onclick="if(event.target.id==='lb')lbClose()"><span class=x onclick="lbClose()">&times;</span>
<div class=nav id=lbprev onclick="lbStep(-1)">&#8249;</div><img id=lbimg src=""><div class=nav id=lbnext onclick="lbStep(1)">&#8250;</div>
<div class=cap id=lbcap></div></div>
<script>
const IMGS={{flat|tojson}};let lbi=0;
function esc(s){return (s||'').replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
function lbShow(){var m=IMGS[lbi];document.getElementById('lbimg').src='/img/'+encodeURIComponent(m.file);
 document.getElementById('lbcap').innerHTML='<span class=n>'+(lbi+1)+' / '+IMGS.length+'</span> · '+
 (m.prompt?esc(m.prompt.slice(0,90))+' · ':'')+'seed '+m.seed+' · <a href="/result/'+m.rid+'">view batch &rarr;</a>';}
function lbOpen(i){lbi=i;lbShow();document.getElementById('lb').style.display='flex';}
function lbClose(){document.getElementById('lb').style.display='none';}
function lbStep(d){lbi=(lbi+d+IMGS.length)%IMGS.length;lbShow();}
document.addEventListener('keydown',function(e){if(document.getElementById('lb').style.display!=='flex')return;
 if(e.key==='ArrowLeft')lbStep(-1);else if(e.key==='ArrowRight')lbStep(1);else if(e.key==='Escape')lbClose();});
(function(){var x0=0,lb=document.getElementById('lb');
 lb.addEventListener('touchstart',function(e){x0=e.changedTouches[0].clientX;},{passive:true});
 lb.addEventListener('touchend',function(e){var dx=e.changedTouches[0].clientX-x0;if(Math.abs(dx)>40)lbStep(dx<0?1:-1);},{passive:true});})();
// After a delete we land on /gallery#b-<rid>: scroll to that batch and flash it.
(function(){var h=location.hash;if(h.indexOf('#b-')!==0)return;
 var el=document.getElementById(h.slice(1));if(!el)return;
 // thumbnails reserve height via aspect-ratio, so layout is stable before load
 el.scrollIntoView({block:'center'});el.classList.add('flash-batch');
 setTimeout(function(){el.classList.remove('flash-batch');},1600);})();
</script>""".replace("{{css}}", BASE_CSS)

if __name__ == "__main__":
    port = int(os.environ.get("F2K_WEB_PORT", "5000"))
    print(f"F2K web UI on http://0.0.0.0:{port}  (user={USER})")
    app.run(host="0.0.0.0", port=port, threaded=True)
