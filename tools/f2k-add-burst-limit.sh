#!/usr/bin/env bash
#
# f2k-add-burst-limit.sh — add a restart burst limit to the F2K units.
#
# Run on spark-f42d as root:
#
#     sudo bash tools/f2k-add-burst-limit.sh
#
# Adds a drop-in to f2k-worker and f2k-octane-bridge so that
# Restart=always gives up eventually instead of retrying every 3 seconds
# forever. Their own unit files are not touched.
#
# 30 attempts in 600 s. With RestartSec=3 that is roughly 90 seconds of
# solid retrying before the unit stops and shows `failed` — far more
# than a cold-boot CUDA race needs (~16 s when it happened on
# spark-dd2f), so a normal boot will not trip it.
#
# The trade: once a unit gives up it stays down until someone runs
#     systemctl reset-failed <unit> && systemctl start <unit>
# A fault that would have self-healed after five minutes no longer does.
# That is the point of a burst limit, but it is a real change.
#
# Safe to re-run. Only daemon-reload is performed, so nothing restarts
# and image generation keeps working while this runs.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "error: run this with sudo — it writes under /etc/systemd/system" >&2
    exit 1
fi

UNITS=(f2k-worker f2k-octane-bridge)

echo "== before =="
for u in "${UNITS[@]}"; do
    printf '  %-22s %-8s burst=%s interval=%s\n' "$u" \
        "$(systemctl is-active "$u" 2>/dev/null || true)" \
        "$(systemctl show "$u" -p StartLimitBurst --value 2>/dev/null || echo '?')" \
        "$(systemctl show "$u" -p StartLimitIntervalUSec --value 2>/dev/null || echo '?')"
done

for u in "${UNITS[@]}"; do
    if ! systemctl cat "$u" >/dev/null 2>&1; then
        echo "warning: no unit named $u — skipping" >&2
        continue
    fi
    dir="/etc/systemd/system/${u}.service.d"
    mkdir -p "$dir"
    cat > "$dir/limits.conf" <<'CONF'
[Unit]
# Give up rather than hammer forever. With RestartSec=3 this is about
# 90 seconds of continuous retrying — ample for a cold-boot CUDA race —
# after which the unit stops and shows `failed` instead of filling the
# journal indefinitely.
#
# Recover with:  systemctl reset-failed UNIT && systemctl start UNIT
StartLimitIntervalSec=600
StartLimitBurst=30
CONF
    echo "wrote $dir/limits.conf"
done

systemctl daemon-reload
echo "daemon-reload done (nothing restarted)"

echo "== after =="
for u in "${UNITS[@]}"; do
    printf '  %-22s %-8s burst=%s interval=%s\n' "$u" \
        "$(systemctl is-active "$u" 2>/dev/null || true)" \
        "$(systemctl show "$u" -p StartLimitBurst --value 2>/dev/null || echo '?')" \
        "$(systemctl show "$u" -p StartLimitIntervalUSec --value 2>/dev/null || echo '?')"
done

echo
echo "If burst is still 0/unset above, the drop-in did not apply —"
echo "check: systemctl cat ${UNITS[0]}"
