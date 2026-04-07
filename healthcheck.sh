#!/bin/bash
# Toolforge health check script for WikiVisage background workers.
#
# Usage: ./healthcheck.sh <worker-id>
#   e.g. --health-check-script "./healthcheck.sh ml-worker-1"
#
# Each worker touches $HOME/.wikivisage-worker-alive-<worker-id> at every
# heartbeat (~60 s). This script checks whether that file exists and was
# modified within the last 5 minutes.
#
# Toolforge liveness probes run every 10 s and restart the job after
# 3 consecutive failures. Because we check file *age* (not existence-
# then-delete), a healthy worker that writes every 60 s will never
# trigger a false restart.

WORKER_ID="${1:?Usage: $0 <worker-id>}"
HEARTBEAT_FILE="$HOME/.wikivisage-worker-alive-${WORKER_ID}"
MAX_AGE_MINUTES=5

# File missing → worker never started or crashed hard
[ -f "$HEARTBEAT_FILE" ] || exit 1

# File older than MAX_AGE_MINUTES → worker is stuck
if find "$HEARTBEAT_FILE" -mmin +$MAX_AGE_MINUTES 2>/dev/null | grep -q .; then
    exit 1
fi

exit 0
