# F2K Spark Admin Runbook

Quick reference for running the three services on the DGX Spark (`spark-f42d`)
after a reboot or when something falls over.

## The three services

| Service | Binary / script | Port | Bound to | Purpose |
|---|---|---|---|---|
| **Worker** | `build/serve` | `8765` | `127.0.0.1` (localhost only) | Persistent engine; models resident → ~8s/image. Everything else talks to this. |
| **Web UI** | `tools/webui/app.py` (Flask) | `5000` | `0.0.0.0` (tailnet) | Browser UI. iPad/laptop → http://spark-f42d:5000 |
| **Octane bridge** | `build/octane_bridge` | `1974` | `0.0.0.0` (LAN) | Protocol translator for the SGI Octane (Roadrunner) client. |

Dependency chain: **Web UI → Worker** and **Bridge → Worker**. The worker must be
up first; the other two are just front-ends that connect to it on `:8765`.

---

## Health check — is everything up?

```bash
# the authoritative check — all three should say "active" (no sudo needed)
systemctl is-active f2k-worker f2k-web f2k-octane-bridge

# listening ports (expect 8765 on loopback, 5000 + 1974 on 0.0.0.0)
ss -tlnp | grep -E ":8765|:5000|:1974"
```

Expected: three `active`, three listening ports. If one is down, `systemctl status
<unit>` says why; start it with the commands below.

A service that is `active` but whose port is missing is still loading — the worker
takes ~12 s to bring up the klein-4B encoder.

---

## Starting things

### Everything the easy way (worker + web UI)

```bash
cd ~/Documents/Development/Cpp/Projects/F2KTest
F2K_WEB_PASSWORD=rocket tools/webui/start.sh
```

`start.sh` is idempotent: it only launches the worker if it isn't already running
(checks `pgrep -f "build/serve --port 8765"`), waits for it to log
`ready, listening`, then starts Flask. Login user defaults to `chris`.

### Worker only (no web UI)

```bash
cd ~/Documents/Development/Cpp/Projects/F2KTest
./build/serve --port 8765 --model ~/models/flux2-klein-4B \
    >/tmp/f2k_serve.log 2>&1 &
# wait until it logs "ready, listening" (~12s to load the klein-4B encoder)
```

`--model` preloads the default model's encoder at boot so the first request is fast.

### Octane bridge (for Roadrunner)

Normally already running (it's a systemd unit — see *Post-reboot notes*). To bring
it back:

```bash
sudo systemctl restart f2k-octane-bridge
journalctl -u f2k-octane-bridge -n 20 --no-pager
```

Or by hand, for debugging in the foreground (stop the unit first so the two don't
fight over `:1974`):

```bash
cd ~/Documents/Development/Cpp/Projects/F2KTest
sudo systemctl stop f2k-octane-bridge
./build/octane_bridge --port 1974 --worker-port 8765 \
    --model flux2-klein-4B --precision fp8
```

On success it logs `listening on 0.0.0.0:1974 -> worker 127.0.0.1:8765` and prints
the list of discovered models (6 as of this writing). Port 1974 = SIGGRAPH's
founding year. The worker does **not** need to be up for the bridge to start — only
to serve an actual request.

Bridge flags:
- `--port` — where Roadrunner connects (default `1974`)
- `--worker-port` — the worker to forward to (default `8765`)
- `--worker-host` — worker host (default `127.0.0.1`)
- `--model` — default model if the client doesn't pick one
- `--precision` — `fp8` (sharp) or `fp4`

---

## Logs

```bash
tail -f /tmp/f2k_serve.log        # worker
tail -f /tmp/octane_bridge.log    # bridge
# web UI logs to the terminal running start.sh (or its nohup/systemd sink)
```

---

## Stopping / restarting a service

```bash
pkill -f "build/serve --port 8765"       # worker  (kills web UI + bridge's backend!)
pkill -f "octane_bridge"                 # bridge
pkill -f "tools/webui/app.py"            # web UI
```

Then re-run the relevant start command above. If you restart the **worker**, the
web UI and bridge stay up but their next request reconnects — no need to restart
them unless they were mid-request.

---

## Post-reboot notes

**All three services come back automatically.** They are `systemd` **system**
units, enabled and started at boot — no login required:

| Service | Unit | Unit file |
|---|---|---|
| Worker | `f2k-worker.service` | `/etc/systemd/system/f2k-worker.service` |
| Web UI | `f2k-web.service` | `/etc/systemd/system/f2k-web.service` |
| Octane bridge | `f2k-octane-bridge.service` | `/etc/systemd/system/f2k-octane-bridge.service` |

The bridge unit is version-controlled at `tools/systemd/f2k-octane-bridge.service`;
install or update it with:

```bash
sudo install -m 644 tools/systemd/f2k-octane-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl restart f2k-octane-bridge.service
```

The bridge only *soft*-depends on the worker (`Wants=`, not `Requires=`): it binds
`:1974` and discovers models from `~/models` unaided, and needs `:8765` only when a
request actually arrives. So boot order between the two doesn't matter, and
restarting the worker never requires restarting the bridge.

### Managing them

```bash
systemctl status  f2k-worker f2k-web f2k-octane-bridge   # no sudo needed to read
sudo systemctl restart f2k-octane-bridge                 # restart one
sudo systemctl disable f2k-octane-bridge                 # stop starting at boot
journalctl -u f2k-octane-bridge -f                       # follow its log
```

All three are `Restart=always`, so a crashed service comes back on its own after
3 s. Prefer `systemctl restart` over the `pkill` + re-launch recipes above — a
hand-started process is invisible to systemd, and systemd will fight it for the
port on the next restart.

> **History:** before 2026-08-19 the bridge had no unit at all, which is why a
> reboot (or Rocket vs. the UPS cable) brought back the worker and web UI but left
> the Octane dark.

### Stale user units — do not re-enable

`~/.config/systemd/user/` also holds `f2k-worker.service` and `f2k-webui.service`,
duplicating the system units above. **Both are now disabled and should stay that
way.** They were shadowed and broken:

- the user `f2k-worker` could never bind `:8765` (the system worker owns it), so it
  crash-looped every 3 s indefinitely;
- the user `f2k-webui` declared both `After=default.target` and
  `WantedBy=default.target`, an ordering cycle systemd broke at every boot by
  silently dropping the job — so it never actually ran.

The files are left on disk for reference only. If you ever want a user-scoped
service, drop the `After=default.target` line and make sure the corresponding
system unit is disabled first, or the two will race for the port.

---

## Worker watchdog (wedge detection)

`Restart=always` only catches a worker that **exits**. It cannot see a worker that
is alive but **wedged**, so the watchdog probes for that case.

> **The original wedge is now fixed in the worker itself** (2026-08-19). It was:
> `serve.cu` is single-threaded and strictly serial (one `accept()`, the whole job
> inline, then `close`), and `read_line()` had no recv timeout, so a client that
> connected and never sent a `\n` blocked the accept loop **forever** — an iPad
> dropping off Tailscale mid-request did exactly this. The process stayed alive and
> `:8765` stayed bound, so `systemctl is-active` reported `active` while every
> image timed out. Accepted sockets now carry `SO_RCVTIMEO`/`SO_SNDTIMEO`
> (`--io-timeout`, default 15 s) and a stalled peer is dropped. See *Client I/O
> timeout* below.

The watchdog is kept as a **backstop**: the socket timeout closes the known hole,
but a CUDA hang or deadlock inside `generate()` would still leave the worker alive
and unable to reach `accept()`, and only a probe can see that.

`tools/f2k_watchdog.sh` closes that gap. Install it with:

```bash
tools/install_watchdog.sh              # asks for sudo itself; safe to re-run
tools/install_watchdog.sh --uninstall  # stop, disable, remove
```

Re-run it to deploy edits to `f2k_watchdog.sh` or its units. It installs the script
to root-owned `/usr/local/sbin`, points the unit there, enables the timer, then runs
one probe against the live worker to prove it works. Afterwards:

```bash
systemctl list-timers f2k-watchdog --no-pager        # next fire time
journalctl -u f2k-watchdog -f                        # watch it work
```

### How it decides

Every 2 minutes it sends a probe and, only on a bad result, escalates:

1. **Unit not `active`** (or restarting, or started < 90 s ago and still doing its
   ~12 s model load) → no-op. systemd owns those cases.
2. **Probe** — a job whose `model` is a nonexistent path. `generate()` fails at
   `ensure_encoder()` and returns **before any GPU work**: ~1 ms, no allocation,
   and it provably never evicts the resident encoder or cached pipeline. A reply
   proves the accept loop, JSON parse and dispatch are all alive — precisely what a
   wedge takes out. Any reply → healthy, counter reset.
3. **No reply, GPU busy** (`utilization.gpu` ≥ 5%) → **hold**. A worker mid-batch
   cannot answer either, because it only reaches `accept()` between jobs; a 9B
   1024px batch runs into minutes and looks identical from the socket.
4. **No reply, GPU idle** → strike. After **3 consecutive** strikes (~6 min) it logs
   the worker's last 10 journal lines and runs `systemctl restart f2k-worker`.

The bias is deliberate and one-directional: **restarting mid-generation destroys an
in-flight image, so every ambiguity resolves toward leaving the worker alone.** If
`nvidia-smi` gives no reading, it holds rather than restarting blind — it will miss
a wedge before it will kill live work. Consequence to know: if something *else*
(e.g. `ollama`) is hammering the GPU, a genuine wedge goes undetected until that
quiets down.

Tunables, via `Environment=` in `f2k-watchdog.service`:

| var | default | meaning |
|---|---|---|
| `F2K_WD_FAILS` | `3` | consecutive strikes before restarting |
| `F2K_WD_PROBE_TIMEOUT` | `20` | seconds to wait for a reply |
| `F2K_WD_BUSY_PCT` | `5` | GPU % that counts as "working, not wedged" |
| `F2K_WD_MIN_UPTIME` | `90` | grace period after a start |

Change the interval in `f2k-watchdog.timer` (`OnUnitActiveSec=`). Only one probe is
ever outstanding (systemd won't re-trigger a `oneshot` still running), which
matters because the worker's `listen()` backlog is only 4.

> **A note on running it as root.** The watchdog needs root to `systemctl restart`
> the worker, and as written it executes a script from `~/Documents/...`, a path
> your normal user can write. On a single-admin box that is unremarkable, but it is
> the only root-run code here loaded from user-writable storage — anyone who can
> write that file gets root. (`f2k-worker`/`f2k-web` don't have this property: they
> run `User=chris`.) To close it, install the script to root-owned storage and
> point the unit there:
>
> ```bash
> sudo install -m 755 -o root -g root tools/f2k_watchdog.sh /usr/local/sbin/f2k_watchdog.sh
> sudo sed -i 's#^ExecStart=.*#ExecStart=/usr/local/sbin/f2k_watchdog.sh#' \
>     /etc/systemd/system/f2k-watchdog.service
> sudo systemctl daemon-reload
> ```
>
> The repo copy then becomes the source of truth you edit, re-running the `install`
> to deploy.

---

## Client I/O timeout (`--io-timeout`)

The worker applies an **idle** timeout to every accepted socket, both directions:

```bash
build/serve --port 8765 --io-timeout 15    # seconds; 0 disables (old behaviour)
```

- **Receive** — a peer that connects and then goes silent is dropped instead of
  blocking the serial accept loop forever (`[worker] client stalled mid-request;
  dropped after 15s`). This is the bug described under *Worker watchdog*.
- **Send** — a peer that stops *reading* can no longer wedge the worker in
  `write()` mid-stream either. Once a write fails the worker stops emitting
  progress for that job, so one dead client cannot burn a timeout per denoise step.

It is **idle-only**: any byte arriving resets it, so a slow-but-live client is
never cut off. Every front-end sends its one short JSON line immediately after
connecting (both pass images as file *paths*, never inline), so real requests are
sub-second.

Two bounds set the 15 s default. It is also the worst-case delay a stalled peer can
inflict on the next client, since the loop is serial — so shorter is better. And it
must stay **below** the watchdog's 20 s probe timeout (`F2K_WD_PROBE_TIMEOUT`), or
a stall being reaped normally would cost the watchdog a spurious strike. Keep that
ordering if you change either number.

---

## Troubleshooting

- **Roadrunner / a client can't connect to the bridge** — first check the bridge is
  running and listening on `:1974` (health check above). If it is, the issue is the
  network path (LAN/firewall between Roadrunner and the Spark), not the bridge.
- **Web UI loads but images error / hang** — the worker (`:8765`) is down or still
  loading. Check `journalctl -u f2k-worker -n 20` for `ready, listening`.
- **Worker says `active` but every image hangs** — the worker is wedged. Since the
  `--io-timeout` fix a half-open client can no longer cause this, so suspect a
  deeper hang (CUDA deadlock in `generate()`). The watchdog clears it within ~6 min;
  to fix it now, `sudo systemctl restart f2k-worker`. Check
  `journalctl -u f2k-watchdog -n 20` (look for `strike N/3`) and
  `journalctl -u f2k-worker -n 30` for what it was doing.
- **One request was slow, then everything was fine** — a peer stalled and the worker
  waited out its 15 s idle timeout before serving the next client. Look for
  `client stalled mid-request` in `journalctl -u f2k-worker`. Expected behaviour, not
  a fault; lower `--io-timeout` if it happens often.
- **Bridge starts but shows no models** — model dirs under `~/models/` are missing
  or renamed; the bridge mirrors the web UI's model discovery.
- **Port already in use on start** — a stale process is holding it; `pkill` it
  (above) and retry.
