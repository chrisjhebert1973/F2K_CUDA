#!/usr/bin/env bash
# Launch the F2K web UI (v2): start the persistent worker if it isn't already
# running, then the Flask front-end. Reach it from any tailnet device at
# http://spark-f42d:5000  (login user defaults to 'chris').
#
#   F2K_WEB_PASSWORD=yourpw tools/webui/start.sh
set -e
cd "$(dirname "$0")/../.."   # repo root

PORT="${F2K_WORKER_PORT:-8765}"
if ! pgrep -f "build/serve --port $PORT" >/dev/null 2>&1; then
    echo "[start] launching persistent worker (resident model load ~12s)..."
    ./build/serve --port "$PORT" >/tmp/f2k_serve.log 2>&1 &
    # wait until it's listening (so the first request hits the fast path)
    for _ in $(seq 1 40); do grep -q "ready, listening" /tmp/f2k_serve.log 2>/dev/null && break; sleep 1; done
    echo "[start] worker: $(tail -1 /tmp/f2k_serve.log)"
else
    echo "[start] worker already running on :$PORT"
fi

export F2K_WEB_PASSWORD="${F2K_WEB_PASSWORD:-rocket}"
echo "[start] web UI → http://spark-f42d:5000  (user=${F2K_WEB_USER:-chris})"
exec .venv/bin/python tools/webui/app.py
