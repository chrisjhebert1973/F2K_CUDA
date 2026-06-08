# F2K_CUDA Web UI — Operations Guide

A personal web front-end for the FLUX.2-klein image generator, reachable from any
device on your Tailscale network (e.g. an iPad). It drives a **persistent worker**
that keeps the models resident, so each image costs ~8 s instead of ~25 s.

```
 iPad (Safari) ──Tailscale──▶  Spark:5000  Flask UI (login, form, gallery)
                                   │ localhost TCP :8765 (JSON)
                                   ▼
                              build/serve  persistent worker (models resident)
                                   │ falls back to one-shot `build/generate` if the
                                   ▼ worker is down (slower, ~25 s)
                              runs/<id>.png  +  <id>.json (prompt, params, timing)
```

---

## 1. Accessing it (from the iPad / any tailnet device)

Open Safari and go to:

```
http://100.80.184.6:5000          ← raw Tailscale IP (most reliable)
http://spark-f42d:5000            ← only if MagicDNS is enabled
```

Log in with your username (default `chris`) and the password you set (see §4).

> Use **`http://`**, not `https://` — Safari likes to auto-upgrade; type the full
> URL. Tailscale (WireGuard) already encrypts the traffic, so plain HTTP is fine.

Once in: type a prompt, choose resolution / precision / steps / seed (blank seed =
random), tap **Generate**. ~8 s later the image appears (tap-and-hold to save it to
the iPad). Every result is kept in the **Gallery**.

---

## 2. Prerequisites (one-time)

- The project is built: `build/serve` and `build/generate` exist
  (`cmake --build build -j` from the repo root).
- The converted models exist under `~/models/flux2-klein-9B/`
  (`transformer_mxfp8/`, `transformer_f2k/`, `qwen3_f2k/`, `vae_f2k/`, `tokenizer/`).
- Flask is in the project venv: `.venv/bin/python -c "import flask"` succeeds
  (`.venv/bin/pip install flask` if not).

Repo root referenced below:
`/home/chris/Documents/Development/Cpp/Projects/F2KTest`

---

## 3. Running it

### A) Quick / manual (foreground, for testing)

```bash
cd ~/Documents/Development/Cpp/Projects/F2KTest
F2K_WEB_PASSWORD='your-password' tools/webui/start.sh
```

`start.sh` launches the persistent worker (if not already running), waits ~12 s for
it to load the resident models, then starts the Flask UI. Press `Ctrl-C` to stop the
UI (the worker keeps running in the background). This does **not** survive a reboot —
for that, use the systemd setup below.

### B) Auto-start on boot (recommended) — systemd

Installs two services: `f2k-worker` (resident models) and `f2k-web` (the Flask UI,
which depends on the worker).

```bash
cd ~/Documents/Development/Cpp/Projects/F2KTest

# set your login password (root-readable only)
mkdir -p ~/.config
echo 'F2K_WEB_PASSWORD=your-strong-password' > ~/.config/f2k-web.env
chmod 600 ~/.config/f2k-web.env
# (optional: add a line  F2K_WEB_USER=chris  to change the username)

# install + enable for boot
sudo cp tools/webui/f2k-worker.service tools/webui/f2k-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now f2k-worker f2k-web
```

> If a manual instance (§A) is still running it holds ports 5000/8765, so
> `--now` can't bind. Either stop it first
> (`pkill -f tools/webui/app.py; pkill -f 'build/serve --port'`) or just
> `sudo reboot` — on reboot the manual instance is gone and systemd starts both
> services cleanly.

After this, the UI starts automatically on every boot. Give it ~15 s after boot for
the worker to load the models (until then the UI still works via the slower
subprocess fallback).

---

## 4. Common operations

All `systemctl`/`journalctl` commands assume the systemd setup (§3B).

| Task | Command |
|---|---|
| **Status** | `systemctl status f2k-worker f2k-web --no-pager` |
| **Start** | `sudo systemctl start f2k-worker f2k-web` |
| **Stop** | `sudo systemctl stop f2k-web f2k-worker` |
| **Restart** | `sudo systemctl restart f2k-worker f2k-web` |
| **Restart just the UI** | `sudo systemctl restart f2k-web` |
| **Watch worker logs** | `journalctl -u f2k-worker -f` |
| **Watch UI logs** | `journalctl -u f2k-web -f` |
| **Disable auto-start** | `sudo systemctl disable --now f2k-worker f2k-web` |

### Change the password

```bash
nano ~/.config/f2k-web.env          # edit F2K_WEB_PASSWORD=...
sudo systemctl restart f2k-web      # picks up the new password
```

### After rebuilding the binary (e.g. a kernel change)

```bash
cmake --build build -j
sudo systemctl restart f2k-worker f2k-web   # reload the resident models
```

### Manual stop/start (if NOT using systemd)

```bash
# stop
pkill -f tools/webui/app.py            # the Flask UI
pkill -f 'build/serve --port'          # the worker
# start (foreground)
F2K_WEB_PASSWORD='your-password' tools/webui/start.sh
```

---

## 5. How it performs

| | time |
|---|---|
| Worker model load (once, at boot/start) | ~12 s |
| Image, same res+precision (worker resident) | **~8 s** (1024px) |
| First image after changing res or precision | ~14 s (rebuilds transformer+VAE; Qwen stays resident) |
| Image with the worker **down** (subprocess fallback) | ~25 s (reloads everything) |

The result page shows which path was used (`worker` vs `subprocess (worker down)`).

---

## 6. Troubleshooting

**iPad can't load the page**
1. Use the raw IP and `http://`: **`http://100.80.184.6:5000`**.
2. Confirm the Tailscale app on the iPad is connected.
3. On the Spark, confirm it's listening:
   `ss -ltnp | grep -E ':5000|:8765'` (expect Flask on `0.0.0.0:5000`, worker on
   `127.0.0.1:8765`).
4. Firewall: `sudo ufw status`. If active and 5000 isn't allowed:
   `sudo ufw allow in on tailscale0` (or `sudo ufw allow 5000/tcp`).

**Every image is slow (~25 s) / result says "subprocess (worker down)"**
The worker isn't running. `systemctl status f2k-worker`; `journalctl -u f2k-worker -e`
for why (e.g. missing model files). `sudo systemctl restart f2k-worker`.

**"bad res" / generation error**
Resolution must satisfy `res % 16 == 0` and `(res/16)² % 128 == 0` — the form's
choices (256 / 512 / 768 / 1024) all qualify.

**Port already in use on start**
A previous instance is still up. Stop it (§4 "Manual stop/start") or reboot.

---

## 7. Files

| Path | What |
|---|---|
| `tools/webui/app.py` | Flask UI (login, form, gallery; talks to the worker, subprocess fallback) |
| `tools/serve.cu` → `build/serve` | persistent worker (resident models, TCP JSON on :8765) |
| `tools/webui/start.sh` | launches worker (if down) + Flask |
| `tools/webui/f2k-worker.service` | systemd unit for the worker |
| `tools/webui/f2k-web.service` | systemd unit for the Flask UI |
| `~/.config/f2k-web.env` | your password / username (you create this) |
| `tools/webui/runs/` | generated images + JSON metadata (the gallery) |

---

## 8. Security notes

This is the Flask **development server**, intended for **personal use behind
Tailscale only** — do not expose it to the public internet (don't `tailscale funnel`
it). The single-user login is a convenience; the real boundary is your tailnet.
Generation args are passed to the binary as an argument list (no shell), so prompts
can't inject commands.
