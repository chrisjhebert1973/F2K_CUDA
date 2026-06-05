#!/usr/bin/env bash
# Run f2k under Xvfb (virtual framebuffer) and grab a screenshot. Used for
# smoke-testing the UI without a real graphical session.
#
# Required packages (one-time):
#   sudo apt install xvfb imagemagick
#
# Usage:
#   scripts/run-headless.sh                # uses ./build/f2k, default screen
#   scripts/run-headless.sh /tmp/shot.png  # custom screenshot path
#
# Caveat: Xvfb is software-only. The NVIDIA Vulkan ICD may refuse to present
# to an Xvfb-backed surface. If you see "no suitable physical device", you'll
# need a real display session (SSH X-forwarding, VNC, or a local desktop).

set -euo pipefail

DISPLAY_NUM="${F2K_DISPLAY:-:99}"
SCREEN="${F2K_SCREEN:-1600x1000x24}"
BINARY="${F2K_BINARY:-./build/f2k}"
SHOT="${1:-/tmp/f2k-screenshot.png}"
WAIT_SECS="${F2K_WAIT_SECS:-2}"

if ! command -v Xvfb >/dev/null; then
    echo "Xvfb not installed. Run: sudo apt install xvfb" >&2
    exit 1
fi
if ! command -v import >/dev/null; then
    echo "ImageMagick 'import' not installed. Run: sudo apt install imagemagick" >&2
    exit 1
fi
if [[ ! -x "$BINARY" ]]; then
    echo "binary not found or not executable: $BINARY" >&2
    exit 1
fi

# Start Xvfb. -nolisten tcp keeps it local-only.
Xvfb "$DISPLAY_NUM" -screen 0 "$SCREEN" -nolisten tcp &
XVFB_PID=$!
cleanup() { kill "$XVFB_PID" 2>/dev/null || true; }
trap cleanup EXIT

sleep 0.5

# Launch the app.
DISPLAY="$DISPLAY_NUM" "$BINARY" &
APP_PID=$!

# Let it draw a frame or two.
sleep "$WAIT_SECS"

if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "f2k exited before screenshot. Check stderr above." >&2
    exit 1
fi

DISPLAY="$DISPLAY_NUM" import -window root "$SHOT"

kill "$APP_PID" 2>/dev/null || true
wait "$APP_PID" 2>/dev/null || true

echo "screenshot: $SHOT"
