#!/usr/bin/env bash
# eval_cleanup_watchdog.sh — keep evaluation_results_*_4omini.json in sync
# with their underlying answer_results_*.json during a fresh reproduction
# run.
#
# Why this exists
# ---------------
# The canonical run_judge.sh has a "is this trial judged?" gate that is
# satisfied purely by the existence of the corresponding
# evaluation_results_*_4omini.json file. When the runner regenerates an
# answer file, the stale paper-baseline eval file underneath still satisfies
# that gate, so the runner-triggered post-trial judge silently no-ops and
# the eval stays stale. This watchdog deletes any eval whose underlying
# answer is fresher; the next run_judge.sh run then sees the trial as
# unjudged and produces a fresh eval.
#
# Race-safety with non-atomic writes
# ----------------------------------
# eval/src/io.write_json is open+truncate+write (NOT tmp+rename), so a
# concurrent unlink of a half-written eval is theoretically possible.
# Mitigation: the SETTLE_SECS guard below — we only delete an eval whose
# mtime is at least SETTLE_SECS old. Eval files are tiny (~1 MB JSON);
# a fully written eval lands within milliseconds, so a 60 s guard
# eliminates the window without slowing us down meaningfully.
#
# Usage
# -----
#   bash scripts/reproduce/eval_cleanup_watchdog.sh
#   nohup bash scripts/reproduce/eval_cleanup_watchdog.sh > /tmp/memarena_eval_cleanup.log 2>&1 &
#
# Stop with: pkill -f eval_cleanup_watchdog.sh

set -euo pipefail

RUN_DIR="${RUN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/MASim/runs/l_20260408_111046}"
SETTLE_SECS="${SETTLE_SECS:-60}"   # don't delete an eval less than this old
START_AFTER="${START_AFTER:-2026-04-25 07:19:00}"  # only consider answers fresher than this
SCAN_INTERVAL="${SCAN_INTERVAL:-30}"

START_TS=$(date -d "$START_AFTER" +%s)

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

log "watchdog started; RUN_DIR=$RUN_DIR settle=${SETTLE_SECS}s start_after=$START_AFTER ($START_TS) scan_interval=${SCAN_INTERVAL}s"

while true; do
    now=$(date +%s)
    for be in vanilla oracle rag ; do
        sub=$be ; [ "$be" = "rag" ] && sub=inmem
        for slot in 0_6b 8b llama3b 7b 32b ; do
            ans="$RUN_DIR/eval_results/$sub/answer_results_${be}_${slot}.json"
            evl="$RUN_DIR/eval_results/$sub/evaluation_results_${be}_${slot}_4omini.json"
            [ -f "$ans" ] || continue
            [ -f "$evl" ] || continue
            ans_mtime=$(stat -c %Y "$ans" 2>/dev/null || echo 0)
            evl_mtime=$(stat -c %Y "$evl" 2>/dev/null || echo 0)
            # eval is "stale" if:
            #   (a) the answer is from this run (not from the paper baseline)
            #   (b) the eval is older than the answer (it predates the answer)
            #   (c) the eval has been settled on disk for > SETTLE_SECS
            #       (race-safety against non-atomic write_json)
            if [ "$ans_mtime" -gt "$START_TS" ] \
                && [ "$evl_mtime" -lt "$ans_mtime" ] \
                && [ $((now - evl_mtime)) -gt "$SETTLE_SECS" ] ; then
                log "STALE: $sub/$be/$slot (ans=$ans_mtime evl=$evl_mtime age=$((now - evl_mtime))s) -> rm eval"
                rm -f "$evl"
            fi
        done
    done
    sleep "$SCAN_INTERVAL"
done
