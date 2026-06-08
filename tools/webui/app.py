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
import os, re, json, time, hmac, secrets, subprocess, threading, socket
from functools import wraps
from flask import (Flask, request, session, redirect, url_for, abort,
                   render_template_string, send_from_directory, flash)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERATE = os.path.join(ROOT, "build", "generate")
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(RUNS, exist_ok=True)

USER = os.environ.get("F2K_WEB_USER", "chris")
PASSWORD = os.environ.get("F2K_WEB_PASSWORD", "rocket")
WORKER_HOST = "127.0.0.1"
WORKER_PORT = int(os.environ.get("F2K_WORKER_PORT", "8765"))
RES_CHOICES = ["256", "512", "768", "1024"]
PREC_CHOICES = ["fp8", "nvfp4"]
COUNT_CHOICES = [1, 2, 4, 8]
MAX_COUNT = 8

app = Flask(__name__)
app.secret_key = os.environ.get("F2K_SECRET", secrets.token_hex(16))
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

def _subprocess_call(out_png, prompt, res, precision, steps, seed):
    """Fallback: one-shot generate binary (slow — reloads the model)."""
    args = [GENERATE, "--prompt", prompt, "--res", str(res), "--precision", precision,
            "--steps", str(steps), "--seed", str(seed), "--out", out_png]
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=600)
    log = (proc.stdout or "") + (proc.stderr or "")
    timing = [ln.strip() for ln in log.splitlines()
              if re.search(r"(ready in|Text encode|Denoise loop|VAE decode|Wrote)", ln)]
    return dict(ok=(proc.returncode == 0 and os.path.exists(out_png)), timing=timing,
                error=(log[-1500:] if proc.returncode != 0 else ""))

def run_batch(prompt, res, precision, steps, seeds):
    """Generate len(seeds) images for one prompt, serialised on the GPU lock."""
    rid = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
    images, mode, err = [], "worker", ""
    t0 = time.time()
    with _gpu_lock:
        for i, seed in enumerate(seeds):
            fn = f"{rid}_{i}.png"
            out_png = os.path.join(RUNS, fn)
            payload = dict(prompt=prompt, res=res, precision=precision,
                           steps=steps, seed=seed, out=out_png)
            try:
                r = _worker_call(payload)
            except OSError:
                mode = "subprocess (worker down)"
                r = _subprocess_call(out_png, prompt, res, precision, steps, seed)
            if not r.get("ok") and not err:
                err = r.get("error", "")
            images.append(dict(seed=seed, file=fn, ok=r.get("ok", False),
                               nans=r.get("nans", 0), timing=r.get("timing", [])))
    meta = dict(id=rid, prompt=prompt, res=res, precision=precision, steps=steps,
                count=len(seeds), ok=any(im["ok"] for im in images),
                elapsed=round(time.time() - t0, 1), error=err, mode=mode,
                when=time.strftime("%Y-%m-%d %H:%M"), images=images)
    save_run(meta)
    return meta

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
    return int(res), precision, steps, seeds

# ----------------------------------------------------------------- routes
@app.route("/")
@login_required
def index():
    return render_template_string(INDEX, res_choices=RES_CHOICES, prec_choices=PREC_CHOICES,
                                  count_choices=COUNT_CHOICES, cards=list_images()[:12])

@app.route("/generate", methods=["POST"])
@login_required
def generate():
    prompt = (request.form.get("prompt", "") or "").strip()[:800]
    if not prompt:
        flash("Enter a prompt."); return redirect(url_for("index"))
    res, precision, steps, seeds = _parse_common(request.form)
    meta = run_batch(prompt, res, precision, steps, seeds)
    return redirect(url_for("result", rid=meta["id"]))

@app.route("/result/<rid>")
@login_required
def result(rid):
    m = load_run(rid)
    if not m: abort(404)
    return render_template_string(RESULT, m=m)

@app.route("/delete/<rid>/<int:idx>", methods=["POST"])
@login_required
def delete(rid, idx):
    m = load_run(rid)
    if not m or idx < 0 or idx >= len(m["images"]): abort(404)
    im = m["images"][idx]
    try: os.remove(os.path.join(RUNS, os.path.basename(im["file"])))
    except OSError: pass
    m["images"].pop(idx)
    if m["images"]:
        m["count"] = len(m["images"]); m["ok"] = any(x.get("ok") for x in m["images"])
        save_run(m)
        flash("Deleted one image.")
        return redirect(url_for("result", rid=rid))
    # last image gone — drop the whole run
    try: os.remove(_meta_path(rid))
    except OSError: pass
    flash("Deleted.")
    return redirect(request.form.get("back") or url_for("gallery"))

@app.route("/gallery")
@login_required
def gallery():
    return render_template_string(GALLERY, cards=list_images())

@app.route("/img/<path:fn>")
@login_required
def img(fn):
    return send_from_directory(RUNS, fn)

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
.tile .bar{display:flex;gap:8px;padding:8px}
.tile .bar form{flex:1;margin:0} .tile .bar button{margin:0;padding:9px;font-size:13px;border-radius:9px}
.del{background:#5a2330} .del:active{background:#46101c}
#ov{display:none;position:fixed;inset:0;background:#000b;align-items:center;justify-content:center;
 z-index:9;flex-direction:column;gap:14px;text-align:center;padding:20px}
.spin{width:46px;height:46px;border:5px solid #3b6cf0;border-top-color:transparent;border-radius:50%;
 animation:s 1s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
"""
LOGIN = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · login</title><style>{{css}}</style><div class=wrap>
<header><h1>F2K_CUDA</h1></header>
{% with m=get_flashed_messages() %}{% if m %}<div class=flash>{{m[0]}}</div>{% endif %}{% endwith %}
<form method=post><label>Username</label><input name=user autocapitalize=off autocorrect=off>
<label>Password</label><input name=password type=password>
<button>Sign in</button></form></div>""".replace("{{css}}", BASE_CSS)

INDEX = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · generate</title><style>{{css}}</style>
<div id=ov><div class=spin></div><div>Generating…<br><span class=muted id=ovsub>resident worker · ~8s per image</span></div></div>
<div class=wrap><header><h1>F2K_CUDA <small>image generator</small></h1>
<a href="{{url_for('gallery')}}">Gallery</a></header>
{% with m=get_flashed_messages() %}{% if m %}<div class=flash>{{m[0]}}</div>{% endif %}{% endwith %}
<form method=post action="{{url_for('generate')}}" onsubmit="document.getElementById('ov').style.display='flex'">
<label>Prompt</label><textarea name=prompt placeholder="My dog Rocket, a black and white Akita husky, ..." autofocus></textarea>
<div class=row>
 <div><label>Resolution</label><select name=res>{% for r in res_choices %}<option {{'selected' if r=='1024'}}>{{r}}</option>{% endfor %}</select></div>
 <div><label>Precision</label><select name=precision>{% for p in prec_choices %}<option {{'selected' if p=='fp8'}}>{{p}}</option>{% endfor %}</select></div>
 <div><label>Steps</label><input name=steps type=number value=4 min=1 max=30></div>
</div>
<div class=row>
 <div><label>Batch</label><select name=count onchange="document.getElementById('ovsub').textContent='resident worker · ~8s × '+this.value">{% for c in count_choices %}<option value="{{c}}">{{c}} image{{'s' if c>1}}</option>{% endfor %}</select></div>
 <div><label>Seed <span class=muted>(blank=random)</span></label><input name=seed type=number placeholder=random></div>
</div><button>Generate</button></form>
{% if cards %}<h1 style="font-size:15px;margin:22px 0 8px;color:#aab0bd">Recent</h1>
<div class=grid>{% for c in cards %}<a class=card href="{{url_for('result',rid=c.rid)}}">
<img src="{{url_for('img',fn=c.file)}}"><div class=c>{{c.prompt[:60]}}</div></a>{% endfor %}</div>{% endif %}
</div>""".replace("{{css}}", BASE_CSS)

RESULT = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · result</title><style>{{css}}</style><div class=wrap>
<header><h1><a href="{{url_for('index')}}">← New</a></h1><a href="{{url_for('gallery')}}">Gallery</a></header>
{% with msg=get_flashed_messages() %}{% if msg %}<div class=flash>{{msg[0]}}</div>{% endif %}{% endwith %}
{% if not m.ok %}<div class=flash>Generation failed.<pre style="white-space:pre-wrap;font-size:12px">{{m.error}}</pre></div>{% endif %}
<div class=grid>
{% for im in m.images %}<div class=tile>
 {% if im.ok %}<img src="{{url_for('img',fn=im.file)}}">{% else %}<div class=c style="padding:24px;text-align:center">failed</div>{% endif %}
 <div class=bar>
  <form method=post action="{{url_for('delete',rid=m.id,idx=loop.index0)}}" onsubmit="return confirm('Delete this image?')">
   <button class=del>Delete</button></form>
 </div>
 <div class=c>seed {{im.seed}}{% if im.nans %} · {{im.nans}} nan{% endif %}</div>
</div>{% endfor %}
</div>
<p class=muted>Tap &amp; hold an image to save it to your iPad.</p>
<div class=meta><b>{{m.prompt}}</b><br>
<code>res {{m.res}} · {{m.precision}} · {{m.steps}} steps · {{m.count}} image{{'s' if m.count>1}}</code><br>
<span class=muted>{{m.when}} · {{m.elapsed}}s total · {{m.mode}}</span>
</div></div>""".replace("{{css}}", BASE_CSS)

GALLERY = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · gallery</title><style>{{css}}</style><div class=wrap>
<header><h1><a href="{{url_for('index')}}">← New</a> Gallery</h1><a href="{{url_for('logout')}}">Sign out</a></header>
{% with msg=get_flashed_messages() %}{% if msg %}<div class=flash>{{msg[0]}}</div>{% endif %}{% endwith %}
<div class=grid>{% for c in cards %}<a class=card href="{{url_for('result',rid=c.rid)}}">
<img src="{{url_for('img',fn=c.file)}}"><div class=c>{{c.prompt[:60]}}</div></a>{% endfor %}</div>
</div>""".replace("{{css}}", BASE_CSS)

if __name__ == "__main__":
    port = int(os.environ.get("F2K_WEB_PORT", "5000"))
    print(f"F2K web UI on http://0.0.0.0:{port}  (user={USER})")
    app.run(host="0.0.0.0", port=port, threaded=True)
