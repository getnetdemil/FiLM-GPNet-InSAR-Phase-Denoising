#!/usr/bin/env bash
# Watcher that submits the FAST SLURM array for each remaining variant.
# Usage: nohup setsid bash scripts/watch_submit_jstars_snaphu_fast_20261119.sh <EVAL_TS> <tag> ... &
set -uo pipefail
cd /scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
TS="$1"; shift
for TAG in "$@"; do
    LOG="logs/JSTARS_eval_infer_${TAG}_${TS}.log"
    echo "$(date -u +%FT%TZ) waiting for inference of $TAG ($LOG)"
    until [[ -f "$LOG" ]] && grep -q "Inference complete" "$LOG"; do sleep 60; done
    JOB=$(bash scripts/submit_jstars_snaphu_fast_20261119.sh "$TAG" "$TS")
    echo "$(date -u +%FT%TZ) submitted $TAG (FAST) -> job $JOB"
done
echo "$(date -u +%FT%TZ) all submitted"
