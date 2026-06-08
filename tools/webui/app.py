#!/usr/bin/env python3
"""F2K_CUDA web UI (v1) — a personal Flask front-end for `generate`, meant to be
reached over Tailscale (e.g. from an iPad). Single-user session login; each
request shells out to ./build/generate (model loads per call in v1, ~25s/image).

Run:
  F2K_WEB_PASSWORD=yourpw .venv/bin/python tools/webui/app.py
Then on any tailnet device:  http://spark-f42d:5000

Env:
  F2K_WEB_USER      login username   (default: chris)
  F2K_WEB_PASSWORD  login password   (default: rocket)   <-- change me
  F2K_SECRET        flask secret key (default: random per start)
  F2K_WEB_PORT      port             (default: 5000)
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

app = Flask(__name__)
app.secret_key = os.environ.get("F2K_SECRET", secrets.token_hex(16))
_gpu_lock = threading.Lock()   # serialize GPU jobs (one generate at a time)

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

# ----------------------------------------------------------------- helpers
def list_runs():
    out = []
    for fn in sorted(os.listdir(RUNS), reverse=True):
        if fn.endswith(".json"):
            try: out.append(json.load(open(os.path.join(RUNS, fn))))
            except Exception: pass
    return out

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

def run_generate(prompt, res, precision, steps, seed):
    rid = f"{time.strftime('%Y%m%d_%H%M%S')}_{seed}"
    out_png = os.path.join(RUNS, rid + ".png")
    payload = dict(prompt=prompt, res=res, precision=precision, steps=steps, seed=seed, out=out_png)
    t0 = time.time()
    with _gpu_lock:
        try:
            r = _worker_call(payload); mode = "worker"
        except OSError:
            r = _subprocess_call(out_png, prompt, res, precision, steps, seed); mode = "subprocess (worker down)"
    meta = dict(id=rid, prompt=prompt, res=res, precision=precision, steps=steps, seed=seed,
                ok=r.get("ok", False), elapsed=round(time.time() - t0, 1),
                timing=r.get("timing", []), error=r.get("error", ""), mode=mode,
                when=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(meta, open(os.path.join(RUNS, rid + ".json"), "w"), indent=2)
    return meta

# ----------------------------------------------------------------- routes
@app.route("/")
@login_required
def index():
    return render_template_string(INDEX, res_choices=RES_CHOICES, prec_choices=PREC_CHOICES,
                                  runs=list_runs()[:12])

@app.route("/generate", methods=["POST"])
@login_required
def generate():
    prompt = (request.form.get("prompt", "") or "").strip()[:800]
    if not prompt:
        flash("Enter a prompt."); return redirect(url_for("index"))
    res = request.form.get("res", "1024"); res = res if res in RES_CHOICES else "1024"
    precision = request.form.get("precision", "fp8")
    precision = precision if precision in PREC_CHOICES else "fp8"
    try: steps = max(1, min(30, int(request.form.get("steps", "4"))))
    except ValueError: steps = 4
    seed_in = (request.form.get("seed", "") or "").strip()
    try: seed = int(seed_in) & 0xFFFFFFFF if seed_in else secrets.randbits(32)
    except ValueError: seed = secrets.randbits(32)
    meta = run_generate(prompt, int(res), precision, steps, seed)
    return redirect(url_for("result", rid=meta["id"]))

@app.route("/result/<rid>")
@login_required
def result(rid):
    p = os.path.join(RUNS, rid + ".json")
    if not os.path.exists(p): abort(404)
    return render_template_string(RESULT, m=json.load(open(p)))

@app.route("/gallery")
@login_required
def gallery():
    return render_template_string(GALLERY, runs=list_runs())

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
.row{display:flex;gap:10px;flex-wrap:wrap} .row>div{flex:1;min-width:120px}
button{margin-top:16px;width:100%;padding:14px;border:0;border-radius:12px;
 background:#3b6cf0;color:#fff;font-size:17px;font-weight:600}
button:active{background:#2f57c4} .flash{background:#5a2330;padding:10px;border-radius:8px;margin:8px 0}
img.gen{width:100%;border-radius:12px;display:block;background:#000}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.card{background:#1d2026;border:1px solid #2c3038;border-radius:12px;overflow:hidden}
.card img{width:100%;display:block} .card .c{padding:8px;font-size:12px;color:#aab0bd}
.meta{background:#1d2026;border:1px solid #2c3038;border-radius:12px;padding:12px;font-size:14px}
.meta code{color:#9fe0a0} .muted{color:#8a8f9c;font-size:13px}
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
<div id=ov><div class=spin></div><div>Generating…<br><span class=muted>~25s — the model loads each run in v1</span></div></div>
<div class=wrap><header><h1>F2K_CUDA <small>image generator</small></h1>
<a href="{{url_for('gallery')}}">Gallery</a></header>
{% with m=get_flashed_messages() %}{% if m %}<div class=flash>{{m[0]}}</div>{% endif %}{% endwith %}
<form method=post action="{{url_for('generate')}}" onsubmit="document.getElementById('ov').style.display='flex'">
<label>Prompt</label><textarea name=prompt placeholder="My dog Rocket, a black and white Akita husky, ..." autofocus></textarea>
<div class=row>
 <div><label>Resolution</label><select name=res>{% for r in res_choices %}<option {{'selected' if r=='1024'}}>{{r}}</option>{% endfor %}</select></div>
 <div><label>Precision</label><select name=precision>{% for p in prec_choices %}<option {{'selected' if p=='fp8'}}>{{p}}</option>{% endfor %}</select></div>
 <div><label>Steps</label><input name=steps type=number value=4 min=1 max=30></div>
 <div><label>Seed <span class=muted>(blank=random)</span></label><input name=seed type=number placeholder=random></div>
</div><button>Generate</button></form>
{% if runs %}<h1 style="font-size:15px;margin:22px 0 8px;color:#aab0bd">Recent</h1>
<div class=grid>{% for m in runs %}<a class=card href="{{url_for('result',rid=m.id)}}">
{% if m.ok %}<img src="{{url_for('img',fn=m.id+'.png')}}">{% endif %}
<div class=c>{{m.prompt[:60]}}</div></a>{% endfor %}</div>{% endif %}
</div>""".replace("{{css}}", BASE_CSS)

RESULT = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · result</title><style>{{css}}</style><div class=wrap>
<header><h1><a href="{{url_for('index')}}">← New</a></h1><a href="{{url_for('gallery')}}">Gallery</a></header>
{% if m.ok %}<img class=gen src="{{url_for('img',fn=m.id+'.png')}}">
<p class=muted>Tap & hold the image to save it to your iPad.</p>{% else %}
<div class=flash>Generation failed.<pre style="white-space:pre-wrap;font-size:12px">{{m.error}}</pre></div>{% endif %}
<div class=meta><b>{{m.prompt}}</b><br>
<code>res {{m.res}} · {{m.precision}} · {{m.steps}} steps · seed {{m.seed}}</code><br>
<span class=muted>{{m.when}} · {{m.elapsed}}s total · {{m.mode}}</span>
{% if m.timing %}<br><span class=muted>{% for t in m.timing %}{{t}} · {% endfor %}</span>{% endif %}
</div></div>""".replace("{{css}}", BASE_CSS)

GALLERY = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>F2K · gallery</title><style>{{css}}</style><div class=wrap>
<header><h1><a href="{{url_for('index')}}">← New</a> Gallery</h1><a href="{{url_for('logout')}}">Sign out</a></header>
<div class=grid>{% for m in runs %}{% if m.ok %}<a class=card href="{{url_for('result',rid=m.id)}}">
<img src="{{url_for('img',fn=m.id+'.png')}}"><div class=c>{{m.prompt[:60]}}</div></a>{% endif %}{% endfor %}</div>
</div>""".replace("{{css}}", BASE_CSS)

if __name__ == "__main__":
    port = int(os.environ.get("F2K_WEB_PORT", "5000"))
    print(f"F2K web UI on http://0.0.0.0:{port}  (user={USER})")
    app.run(host="0.0.0.0", port=port, threaded=True)
