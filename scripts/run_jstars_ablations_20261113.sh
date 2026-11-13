#!/usr/bin/env bash
# =============================================================================
# scripts/run_jstars_ablations_20261113.sh
#
# Serial ablation sweep for JSTARS Table II on AOI_000 (Hawaii).
#
# Launches six training variants via the L_tmp-active fork
#   experiments/enhanced/train_film_unet_ltmp_active_20261113.py
# and records per-run metadata to logs/jstars_ablation_log.csv.
#
# Eval (compute_metrics + SNAPHU) is OPT-IN via --with-eval; the default
# is training-only so the sweep stays within the single-GPU budget.
#
# Usage:
#   bash scripts/run_jstars_ablations_20261113.sh --dry-run
#   bash scripts/run_jstars_ablations_20261113.sh                 # train only
#   bash scripts/run_jstars_ablations_20261113.sh --with-eval     # train + eval
#   bash scripts/run_jstars_ablations_20261113.sh --only jstars_abl_no_cls
#
# Design rules honoured (see memory/feedback_script_naming_and_preservation.md):
#   - New script with timestamped filename (`_20261113`).
#   - Calls the forked training script rather than modifying the original.
#   - Idempotent CSV header; `--only` lets you re-run a single variant.
# =============================================================================

set -euo pipefail

# ── Environment (match CLAUDE.md common run prefix) ──
REPO="${REPO:-/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement}"
ENV_PREFIX="${ENV_PREFIX:-/scratch/gdemil24/hrwsi_s3client/torch-gpu}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO}"

# ── CLI parsing ──
DRY_RUN=0
ONLY_TAG=""
WITH_EVAL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --only)      ONLY_TAG="$2"; shift 2 ;;
        --with-eval) WITH_EVAL=1; shift ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Pre-flight checks ──
TRAIN_SCRIPT="experiments/enhanced/train_film_unet_ltmp_active_20261113.py"
DATA_CFG="configs/data/capella_aoi000_ltmp_ablation_20261113.yaml"
MODEL_CFG="configs/model/film_unet.yaml"
TRAIN_CFG="configs/train/contest.yaml"
TRIPLETS="${TRIPLETS:-data/manifests/hawaii_triplets_strict.parquet}"
PAIRS_DIR="${PAIRS_DIR:-data/processed/pairs}"
COP_DEM_DIR="${COP_DEM_DIR:-data/reference/copernicus_dem}"

for f in "$TRAIN_SCRIPT" "$DATA_CFG" "$MODEL_CFG" "$TRAIN_CFG"; do
    if [[ ! -f "$f" ]]; then
        echo "PRE-FLIGHT FAIL: missing $f" >&2; exit 1
    fi
done
if [[ ! -f "$TRIPLETS" ]]; then
    echo "WARN: triplets manifest not found at $TRIPLETS — jstars_abl_no_cls will still run (that variant drops triplets), but the others will exit early."
fi
if [[ ! -d "$PAIRS_DIR" ]]; then
    echo "WARN: pairs dir $PAIRS_DIR not found. Will cause --with-eval to fail."
fi

# ── GPU readiness check (skipped on dry-run) ──
if [[ $DRY_RUN -eq 0 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES}" 2>/dev/null | tr -d ' \n' || echo "0")
        if [[ "${gpu_util:-0}" =~ ^[0-9]+$ ]] && [[ "${gpu_util}" -gt 50 ]]; then
            echo "PRE-FLIGHT FAIL: GPU ${CUDA_VISIBLE_DEVICES} is busy (${gpu_util}%). Aborting." >&2
            exit 1
        fi
    else
        echo "WARN: nvidia-smi not found; skipping GPU readiness check."
    fi
fi

# ── Sweep timestamp (used for all orchestration outputs) ──
SWEEP_TS="${SWEEP_TS:-$(date -u +%Y%m%d_%H%M)}"
CSV="logs/jstars_ablation_log_${SWEEP_TS}.csv"
mkdir -p logs experiments/enhanced/outputs
if [[ ! -f "$CSV" ]]; then
    echo "tag,start,end,duration_h,ckpt_dir,ckpt_path,closure_m1,unwrap_m2,nmad_m4,temporal_m5,eval_csv" > "$CSV"
fi
echo "Sweep CSV: $CSV"

# ── Base training flags (shared by all variants) ──
BASE_FLAGS="--data_config ${DATA_CFG} --model_config ${MODEL_CFG} --train_config ${TRAIN_CFG} --epochs 20"
BASE_WITH_TRI="${BASE_FLAGS} --triplets_manifest ${TRIPLETS}"

# ── Variant matrix: tag -> extra flags.
# Special sentinel __NO_TRIPLETS__ means "use BASE_FLAGS without the triplet
# manifest" (the clean −L_cls ablation; see plan §5).
declare -A VARIANT_FLAGS=(
    [jstars_full_tmp_active]=""
    [jstars_abl_no_unc]="--loss_unc 0"
    [jstars_abl_no_cls]="__NO_TRIPLETS__"
    [jstars_abl_no_tmp]="--loss_temporal 0"
    [jstars_abl_no_grd]="--loss_grad 0"
    [jstars_abl_zero_film]="--zero_film"
)

# Launch order: anchor first; fastest variant second (sanity); rest after.
ORDER=(
    jstars_full_tmp_active
    jstars_abl_no_cls
    jstars_abl_no_unc
    jstars_abl_no_grd
    jstars_abl_no_tmp
    jstars_abl_zero_film
)

PYTHON_RUN=(conda run --prefix "$ENV_PREFIX" --no-capture-output python -u)

# ── Per-variant launcher ──
run_variant() {
    local tag="$1"
    local extra="${VARIANT_FLAGS[$tag]}"
    local flags
    if [[ "$extra" == "__NO_TRIPLETS__" ]]; then
        flags="$BASE_FLAGS"
    else
        flags="$BASE_WITH_TRI $extra"
    fi

    local start_iso start_s end_iso end_s dur_h
    start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_s=$(date -u +%s)

    echo "────────────────────────────────────────────────────────────────────"
    echo "[$tag] start=$start_iso"
    echo "[$tag] flags=$flags"
    echo "[$tag] cmd  = ${PYTHON_RUN[*]} $TRAIN_SCRIPT $flags --run_name $tag"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[$tag] DRY-RUN (not executed)"
        return 0
    fi

    # Run training. We rely on train script writing to logs/<run_tag>.log itself.
    "${PYTHON_RUN[@]}" "$TRAIN_SCRIPT" $flags --run_name "$tag"

    end_s=$(date -u +%s)
    end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    dur_h=$(python3 -c "print(round((${end_s}-${start_s})/3600.0, 2))")

    # Locate the checkpoint the train script just wrote
    local ckpt_dir ckpt
    ckpt_dir=$(ls -dt experiments/enhanced/checkpoints/film_unet/${tag}_* 2>/dev/null | head -1 || true)
    if [[ -z "$ckpt_dir" ]]; then
        echo "[$tag] ERROR: could not locate checkpoint dir matching ${tag}_*"
        echo "$tag,$start_iso,$end_iso,$dur_h,MISSING,MISSING,ERR,ERR,ERR,ERR,MISSING" >> "$CSV"
        return 1
    fi
    ckpt="${ckpt_dir}/$(basename "$ckpt_dir")_final.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "[$tag] ERROR: expected final checkpoint $ckpt not found"
        echo "$tag,$start_iso,$end_iso,$dur_h,$ckpt_dir,MISSING,ERR,ERR,ERR,ERR,MISSING" >> "$CSV"
        return 1
    fi

    # ── Optional eval (M1/M2/M4/M5) ──
    local m1="n/a" m2="n/a" m4="n/a" m5="n/a" eval_csv="n/a"
    if [[ $WITH_EVAL -eq 1 ]]; then
        local out_dir="experiments/enhanced/outputs/${tag}"
        mkdir -p "$out_dir"

        # Step 1: inference + M1/M5
        "${PYTHON_RUN[@]}" eval/compute_metrics.py \
            --checkpoint "$ckpt" \
            --pairs_dir "$PAIRS_DIR" \
            --triplets_manifest "$TRIPLETS" \
            --out_dir "$out_dir" \
            --stride 256 --batch_size 64 \
            --force_inference --skip_snaphu_metrics \
            2>&1 | tee -a "logs/${tag}_eval_${SWEEP_TS}.log" || true

        # Step 2: SNAPHU unwrap of ifg_film_unet.tif (per pair)
        "$ENV_PREFIX/bin/python" -u scripts/unwrap_snaphu.py \
            --pairs_dir "$PAIRS_DIR" \
            --input_ifg ifg_film_unet.tif \
            --output_name unw_phase_film_unet.tif \
            2>&1 | tee -a "logs/${tag}_eval_${SWEEP_TS}.log" || true

        # Step 3: M2 / M3 / M4
        "${PYTHON_RUN[@]}" eval/compute_metrics.py \
            --checkpoint "$ckpt" \
            --pairs_dir "$PAIRS_DIR" \
            --triplets_manifest "$TRIPLETS" \
            --out_dir "$out_dir" \
            --copernicus_dem_dir "$COP_DEM_DIR" \
            --snaphu_only \
            2>&1 | tee -a "logs/${tag}_eval_${SWEEP_TS}.log" || true

        eval_csv="${out_dir}/metrics_comparison.csv"
        # Best-effort metric extraction; leave "n/a" if row not found
        if [[ -f "$eval_csv" ]]; then
            m1=$(awk -F, 'tolower($1) ~ /film.*gpnet|filmunet/ {print $2; exit}' "$eval_csv" || echo "n/a")
            m2=$(awk -F, 'tolower($1) ~ /film.*gpnet|filmunet/ {print $3; exit}' "$eval_csv" || echo "n/a")
            m4=$(awk -F, 'tolower($1) ~ /film.*gpnet|filmunet/ {print $5; exit}' "$eval_csv" || echo "n/a")
            m5=$(awk -F, 'tolower($1) ~ /film.*gpnet|filmunet/ {print $6; exit}' "$eval_csv" || echo "n/a")
            : "${m1:=n/a}"; : "${m2:=n/a}"; : "${m4:=n/a}"; : "${m5:=n/a}"
        fi
    fi

    echo "$tag,$start_iso,$end_iso,$dur_h,$ckpt_dir,$ckpt,$m1,$m2,$m4,$m5,$eval_csv" >> "$CSV"
    echo "[$tag] done: duration=${dur_h}h ckpt=$ckpt closure=$m1 unwrap=$m2 nmad=$m4 temporal=$m5"
}

# ── Main loop ──
for tag in "${ORDER[@]}"; do
    if [[ -n "$ONLY_TAG" && "$tag" != "$ONLY_TAG" ]]; then
        continue
    fi
    run_variant "$tag"
done

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "Ablation sweep complete. Summary:"
echo "────────────────────────────────────────────────────────────────────"
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
