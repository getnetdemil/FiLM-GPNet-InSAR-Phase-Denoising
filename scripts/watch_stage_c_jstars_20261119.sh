#!/usr/bin/env bash
# =============================================================================
# scripts/watch_stage_c_jstars_20261119.sh
#
# Watcher: for each JSTARS eval variant, as soon as all 224 pairs have
# unw_phase_film_unet.tif, run stage C (M2 + M4) on a free CPU slot on the
# login node. Uses compute_metrics_jstars_20261119.py with --snaphu_only
# --skip_inference. Takes ~30 s per variant.
#
# Writes stage-C metrics CSV into
#   experiments/enhanced/outputs/JSTARS_eval_<tag>_<EVAL_TS>/metrics_...csv
# and a stage log
#   logs/JSTARS_eval_m2m4_<tag>_<EVAL_TS>.log
#
# Usage:
#   nohup setsid bash scripts/watch_stage_c_jstars_20261119.sh <EVAL_TS> <tag> ... &
# =============================================================================
set -uo pipefail
cd /scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
ENV_PREFIX=/scratch/gdemil24/hrwsi_s3client/torch-gpu
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement"
PY="${ENV_PREFIX}/bin/python"

TS="$1"; shift
declare -A DONE=()
TRIPLETS="data/manifests/hawaii_triplets_strict.parquet"
COP_DEM_DIR="data/reference/copernicus_dem"
CKPT_ROOT="experiments/enhanced/checkpoints/film_unet"

echo "$(date -u +%FT%TZ) stage-C watcher started, waiting on: $*"

while :; do
    all_done=1
    for TAG in "$@"; do
        [[ "${DONE[$TAG]:-}" == "1" ]] && continue
        TREE="data/processed/JSTARS_eval_${TAG}_${TS}"
        if [[ ! -d "$TREE" ]]; then all_done=0; continue; fi
        n=$(find "$TREE" -mindepth 2 -maxdepth 2 -name unw_phase_film_unet.tif -type f 2>/dev/null | wc -l)
        if [[ "$n" -lt 224 ]]; then all_done=0; continue; fi

        # All 224 unwrapped → run stage C
        ckpt_dir=$(ls -d ${CKPT_ROOT}/${TAG}_2026* 2>/dev/null | grep -v smoke | head -1)
        ckpt="${ckpt_dir}/$(basename "$ckpt_dir")_final.pt"
        if [[ ! -f "$ckpt" ]]; then
            echo "$(date -u +%FT%TZ) $TAG: MISSING ckpt $ckpt"; DONE[$TAG]=1; continue
        fi
        out="experiments/enhanced/outputs/JSTARS_eval_${TAG}_${TS}"
        stage_log="logs/JSTARS_eval_m2m4_${TAG}_${TS}.log"
        echo "$(date -u +%FT%TZ) $TAG: 224/224 unwrapped → running stage C → $stage_log"
        "$PY" -u eval/compute_metrics_jstars_20261119.py \
            --checkpoint "$ckpt" --pairs_dir "$TREE" --triplets_manifest "$TRIPLETS" \
            --copernicus_dem_dir "$COP_DEM_DIR" --out_dir "$out" \
            --stride 256 --batch_size 64 --snaphu_only --skip_inference \
            > "$stage_log" 2>&1
        rc=$?
        echo "$(date -u +%FT%TZ) $TAG: stage C exit=$rc"
        DONE[$TAG]=1
    done
    [[ "$all_done" -eq 1 ]] && break
    sleep 120
done
echo "$(date -u +%FT%TZ) stage-C watcher: all variants done"
