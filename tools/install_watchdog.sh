#!/usr/bin/env bash
# install_watchdog.sh — install (or update) the F2K worker watchdog timer.
#
# Re-runnable: use it to deploy changes to f2k_watchdog.sh or its unit files too.
# Re-execs itself under sudo if needed, so run it as yourself and just enter the
# password once.
#
#   tools/install_watchdog.sh              install / update, then verify
#   tools/install_watchdog.sh --uninstall  stop, disable and remove
#
# The watchdog script is deliberately installed to root-owned /usr/local/sbin
# rather than run out of the repo: it executes as root, and a root-run script
# living in user-writable storage means anyone who can write that file gets root.
# The repo copy stays the source of truth — edit it, re-run this to deploy.

set -euo pipefail

# Resolve the repo from this script's own location, so it works from any cwd.
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
TOOLS="$(dirname "$SELF")"
REPO="$(dirname "$TOOLS")"

SBIN=/usr/local/sbin/f2k_watchdog.sh
UNIT_DIR=/etc/systemd/system
SRC_SCRIPT="$TOOLS/f2k_watchdog.sh"
SRC_SERVICE="$TOOLS/systemd/f2k-watchdog.service"
SRC_TIMER="$TOOLS/systemd/f2k-watchdog.timer"

say() { echo "==> $*"; }
die() { echo "error: $*" >&2; exit 1; }

# Validate arguments BEFORE escalating — no point asking for a password only to
# then reject a typo.
case "${1:-}" in
    ""|--uninstall) ;;
    *) die "unknown argument: $1 (try --uninstall)" ;;
esac
[[ $# -le 1 ]] || die "too many arguments"

# Re-exec under sudo rather than sprinkling sudo through every line: one prompt,
# and the whole install is atomic with respect to the password.
if [[ $EUID -ne 0 ]]; then
    command -v sudo >/dev/null || die "not root and sudo not found"
    say "escalating with sudo..."
    exec sudo -- "$SELF" "$@"
fi

if [[ "${1:-}" == "--uninstall" ]]; then
    say "removing the watchdog"
    systemctl disable --now f2k-watchdog.timer 2>/dev/null || true
    systemctl stop f2k-watchdog.service 2>/dev/null || true
    rm -f "$UNIT_DIR/f2k-watchdog.timer" "$UNIT_DIR/f2k-watchdog.service" "$SBIN"
    rm -rf /run/f2k-watchdog
    systemctl daemon-reload
    say "removed. The worker itself is untouched and still running."
    exit 0
fi

for f in "$SRC_SCRIPT" "$SRC_SERVICE" "$SRC_TIMER"; do
    [[ -r "$f" ]] || die "missing $f — run this from a complete checkout"
done
bash -n "$SRC_SCRIPT" || die "$SRC_SCRIPT has a syntax error; refusing to install it as root"

say "installing $SBIN (root-owned)"
install -m 755 -o root -g root "$SRC_SCRIPT" "$SBIN"

# Point the unit at the installed copy, not the repo one, as it is written out.
say "installing unit files into $UNIT_DIR"
sed 's#^ExecStart=.*#ExecStart='"$SBIN"'#' "$SRC_SERVICE" >"$UNIT_DIR/f2k-watchdog.service"
chmod 644 "$UNIT_DIR/f2k-watchdog.service"
install -m 644 -o root -g root "$SRC_TIMER" "$UNIT_DIR/f2k-watchdog.timer"

say "reloading systemd and enabling the timer"
systemctl daemon-reload
systemctl enable --now f2k-watchdog.timer

# Prove it actually runs, rather than just claiming it is enabled: one probe now,
# against the live worker. A healthy cycle is silent and exits 0.
say "running one probe to verify"
if systemctl start f2k-watchdog.service; then
    say "probe cycle completed"
else
    echo "warning: the probe cycle exited non-zero — check the journal below" >&2
fi

echo
say "status"
systemctl is-enabled f2k-watchdog.timer | sed 's/^/    timer enabled: /'
systemctl is-active  f2k-watchdog.timer | sed 's/^/    timer active:  /'
systemctl list-timers f2k-watchdog --no-pager 2>/dev/null | sed -n '1,2p' | sed 's/^/    /'
echo
say "last watchdog log lines (silence here means healthy)"
journalctl -u f2k-watchdog -n 10 --no-pager 2>/dev/null | sed 's/^/    /'
echo
say "done. Follow it with:  journalctl -u f2k-watchdog -f"
