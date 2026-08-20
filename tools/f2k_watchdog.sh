#!/usr/bin/env bash
# f2k_watchdog.sh — restart the F2K worker if it wedges (alive but not answering).
#
# Why this exists: serve.cu is single-threaded and strictly serial — one accept(),
# the whole job inline, then close. read_line() has NO recv timeout, so a client
# that connects and never sends a '\n' (an iPad dropping off Tailscale mid-request
# is the classic) blocks the accept loop *forever*. The process stays alive, the
# port stays bound, so `systemctl is-active` says "active" while every request
# times out. Restart=always cannot see this; only a probe can.
#
# The probe: an unknown model root makes generate() fail at ensure_encoder() and
# return before ANY GPU work — ~1 ms, no allocation, and (verified) it never
# evicts the resident encoder or the cached pipeline. So probing is free.
#
# The danger is the opposite mistake: restarting mid-generation destroys the
# user's in-flight image. A busy worker cannot answer a probe either (it only
# reaches accept() between jobs), so "no reply" alone is NOT evidence of a wedge.
# We therefore restart only when the probe fails AND the GPU is idle AND that has
# held for FAILS_NEEDED consecutive cycles. Every ambiguity resolves toward
# leaving the worker alone.

set -uo pipefail

UNIT="${F2K_WD_UNIT:-f2k-worker.service}"
PORT="${F2K_WD_PORT:-8765}"
HOST="${F2K_WD_HOST:-127.0.0.1}"
PROBE_TIMEOUT="${F2K_WD_PROBE_TIMEOUT:-20}"   # seconds to wait for a reply
FAILS_NEEDED="${F2K_WD_FAILS:-3}"             # consecutive bad cycles before acting
BUSY_PCT="${F2K_WD_BUSY_PCT:-5}"              # GPU util % that counts as "working"
MIN_UPTIME="${F2K_WD_MIN_UPTIME:-90}"         # grace after start (~12s model load)
STATE_DIR="${RUNTIME_DIRECTORY:-/run/f2k-watchdog}"
STATE="$STATE_DIR/fails"

log() { echo "[watchdog] $*"; }

mkdir -p "$STATE_DIR" 2>/dev/null || true
fails=0
[[ -r "$STATE" ]] && fails=$(<"$STATE") && [[ "$fails" =~ ^[0-9]+$ ]] || fails=0

reset_and_exit() { echo 0 >"$STATE" 2>/dev/null || true; exit 0; }
hold_and_exit()  { exit 0; }   # leave the counter untouched

# 1. Only supervise a unit that is meant to be up and running. "activating" is a
#    normal restart in progress; systemd owns that case, not us.
state=$(systemctl is-active "$UNIT" 2>/dev/null || true)
if [[ "$state" != "active" ]]; then
    log "$UNIT is '$state' — systemd is handling it, nothing to do"
    reset_and_exit
fi

# 2. Grace period: the worker binds its port before a ~12 s model load, so it is
#    legitimately unable to answer for a while after starting.
ts=$(systemctl show -p ActiveEnterTimestampMonotonic --value "$UNIT" 2>/dev/null || echo 0)
if [[ "$ts" =~ ^[0-9]+$ ]] && (( ts > 0 )); then
    now_us=$(awk '{printf "%d", $1 * 1000000}' /proc/uptime 2>/dev/null || echo 0)
    up=$(( (now_us - ts) / 1000000 ))
    if (( up >= 0 && up < MIN_UPTIME )); then
        log "$UNIT started ${up}s ago (< ${MIN_UPTIME}s grace) — still loading, skipping"
        reset_and_exit
    fi
fi

# 3. The probe. Any well-formed reply proves the accept loop, the JSON parse and
#    the dispatch are all alive — which is exactly what a wedge takes out.
if /usr/bin/python3 - "$HOST" "$PORT" "$PROBE_TIMEOUT" <<'PY'
import json, socket, sys
host, port, tmo = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
try:
    s = socket.create_connection((host, port), timeout=tmo)
    s.settimeout(tmo)
    # Unknown model root -> ensure_encoder() fails and returns before any GPU work.
    s.sendall((json.dumps({"model": "/nonexistent/f2k-watchdog-probe"}) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        c = s.recv(65536)
        if not c:
            break
        buf += c
    s.close()
    json.loads(buf.decode("utf-8", "replace").split("\n", 1)[0])   # must be real JSON
except Exception as e:
    print("probe failed: %s: %s" % (type(e).__name__, e), file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PY
then
    if (( fails > 0 )); then
        log "worker answered — recovered after $fails bad cycle(s)"
    fi
    reset_and_exit
fi

# 4. No reply. Before blaming the worker, ask whether the GPU is doing work: a
#    long batch (9B @ 1024 runs into minutes) looks identical from the socket.
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
       | head -1 | tr -dc '0-9')
if [[ -z "$util" ]]; then
    # No usable reading. Deliberately the conservative branch: we would rather
    # miss a wedge than kill a running generation on a blind guess.
    log "no reply, and GPU utilisation unreadable — holding at $fails (will not restart blind)"
    hold_and_exit
fi
if (( util >= BUSY_PCT )); then
    log "no reply, but GPU busy (${util}%) — generating, not wedged; holding at $fails"
    hold_and_exit
fi

# 5. No reply and an idle GPU. That is a wedge candidate.
fails=$(( fails + 1 ))
echo "$fails" >"$STATE" 2>/dev/null || true
log "no reply, GPU idle (${util}%) — strike $fails/$FAILS_NEEDED"

if (( fails < FAILS_NEEDED )); then
    exit 0
fi

log "RESTARTING $UNIT after $fails consecutive unanswered probes with an idle GPU"
log "--- worker journal before restart ---"
journalctl -u "$UNIT" -n 10 --no-pager 2>/dev/null | sed 's/^/[watchdog]   /'
if systemctl restart "$UNIT"; then
    log "restart issued; the front-ends (web UI, Octane bridge) reconnect on their next request"
else
    log "ERROR: systemctl restart $UNIT failed"
fi
echo 0 >"$STATE" 2>/dev/null || true
exit 0
