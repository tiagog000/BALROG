#!/usr/bin/env bash
# Regenerate the capability grids whenever the eval data grows.
# Polls every CHECK_EVERY seconds; only re-runs the (cheap) viz when the number
# of per-episode JSON files under results/1610_1145-* actually changes, so an
# idle queue costs nothing. Logs each regeneration to results/viz/regen.log.
set -u
cd "$(dirname "$0")/../.." || exit 1          # -> BALROG/
CHECK_EVERY="${CHECK_EVERY:-300}"             # 5 min default
LOG=results/viz/regen.log
PY=.venv/bin/python

count_eps() { find results -type d -name '1610_1145-*' -exec find {} -name '*_run_*.json' \; 2>/dev/null | wc -l | tr -d ' '; }

prev=-1
echo "[$(date '+%F %T')] watcher started (every ${CHECK_EVERY}s, pid $$)" >>"$LOG"
while true; do
  cur=$(count_eps)
  if [ "$cur" != "$prev" ]; then
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES "$PY" -I scripts/viz/capability_grid.py \
      >>"$LOG" 2>&1 \
      && echo "[$(date '+%F %T')] regenerated ($prev -> $cur episode files)" >>"$LOG"
    prev=$cur
  fi
  sleep "$CHECK_EVERY"
done
