#!/usr/bin/env bash
# =============================================================================
# scripts/run_jstars_eval_20261119.sh
#
# Evaluate the 6 JSTARS Table II ablation checkpoints WITHOUT touching IGARSS
# outputs. For each variant a fresh tree data/processed/JSTARS_eval_<tag>_<TS>/
# is built with read-only symlinks to data/processed/pairs/ inputs; all eval
# outputs (ifg_film_unet.tif, log_var.tif, unw_phase_film_unet.tif) are written
# as new files inside that tree. Metrics go to
# experiments/enhanced/outputs/JSTARS_eval_<tag>_<TS>/.
#
# Stages (run separately; SNAPHU is the expensive one):
#   --stage infer    : build tree (if missing) + inference + M1 closure + M5 temporal  (GPU)
#   --stage unwrap   : SNAPHU on ifg_film_unet.tif → unw_phase_film_unet.tif          (CPU)
#   --stage m2m4     : M2 unwrap success + M4 height NMAD                              (CPU)
#
# Usage:
#   EVAL_TS=20261119_0900 bash scripts/run_jstars_eval_20261119.sh --stage infer
#   EVAL_TS=20261119_0900 bash scripts/run_jstars_eval_20261119.sh --stage infer --only jstars_abl_no_unc
# The same EVAL_TS must be reused across stages so they share the trees.
# =============================================================================
set -euo pipefail

REPO="${REPO:-/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement}"
ENV_PREFIX="${ENV_PREFIX:-/scratch/gdemil24/hrwsi_s3client/torch-gpu}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${ENV_PREFIX}/bin/python"
cd "$REPO"

STAGE="" ; ONLY="" ; WORKERS="${WORKERS:-2}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --only)  ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
[[ -z "$STAGE" ]] && { echo "--stage {infer|unwrap|m2m4} required"; exit 1; }
EVAL_TS="${EVAL_TS:?set EVAL_TS (YYYYMMDD_HHMM) and reuse it across stages}"

TRIPLETS="data/manifests/hawaii_triplets_strict.parquet"
COP_DEM_DIR="data/reference/copernicus_dem"
CKPT_ROOT="experiments/enhanced/checkpoints/film_unet"

VARIANTS=(
    jstars_full_tmp_active
    jstars_abl_no_unc
    jstars_abl_no_cls
    jstars_abl_no_tmp
    jstars_abl_no_grd
    jstars_abl_zero_film
)

for tag in "${VARIANTS[@]}"; do
    [[ -n "$ONLY" && "$tag" != "$ONLY" ]] && continue

    ckpt_dir=$(ls -d ${CKPT_ROOT}/${tag}_2026* 2>/dev/null | grep -v smoke | head -1)
    ckpt="${ckpt_dir}/$(basename "$ckpt_dir")_final.pt"
    [[ -f "$ckpt" ]] || { echo "[$tag] MISSING checkpoint $ckpt"; continue; }
    tree="data/processed/JSTARS_eval_${tag}_${EVAL_TS}"
    out="experiments/enhanced/outputs/JSTARS_eval_${tag}_${EVAL_TS}"
    zf=""; [[ "$tag" == "jstars_abl_zero_film" ]] && zf="--zero_film"
    stage_log="logs/JSTARS_eval_${STAGE}_${tag}_${EVAL_TS}.log"

    echo "────────────────────────────────────────────────────────────"
    echo "[$tag] stage=$STAGE ckpt=$ckpt tree=$tree $zf  (log: $stage_log)"
    rc=0
    case "$STAGE" in
        infer)
            if [[ ! -d "$tree" ]]; then
                "$PY" scripts/build_jstars_eval_tree_20261119.py --variant "$tag" --ts "$EVAL_TS" >/dev/null
            fi
            "$PY" -u eval/compute_metrics_jstars_20261119.py \
                --checkpoint "$ckpt" --pairs_dir "$tree" --triplets_manifest "$TRIPLETS" \
                --out_dir "$out" --stride 256 --batch_size 64 \
                --force_inference --skip_snaphu_metrics $zf > "$stage_log" 2>&1 || rc=$?
            ;;
        unwrap)
            "$PY" -u scripts/unwrap_snaphu.py --pairs_dir "$tree" \
                --input_ifg ifg_film_unet.tif --output_name unw_phase_film_unet.tif \
                --workers "$WORKERS" > "$stage_log" 2>&1 || rc=$?
            ;;
        m2m4)
            "$PY" -u eval/compute_metrics_jstars_20261119.py \
                --checkpoint "$ckpt" --pairs_dir "$tree" --triplets_manifest "$TRIPLETS" \
                --copernicus_dem_dir "$COP_DEM_DIR" --out_dir "$out" \
                --stride 256 --batch_size 64 --snaphu_only --skip_inference $zf \
                > "$stage_log" 2>&1 || rc=$?
            ;;
        *) echo "bad stage $STAGE"; exit 1 ;;
    esac
    echo "[$tag] stage=$STAGE exit=$rc"
    grep -E "Triplet Closure|Unwrap Success|DEM NMAD|Temporal Residual" "$stage_log" || true
done
echo "Done stage=$STAGE EVAL_TS=$EVAL_TS"
