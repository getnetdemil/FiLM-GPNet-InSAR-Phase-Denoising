#!/usr/bin/env bash
# Submit the itee-partition SNAPHU SLURM array for one JSTARS eval variant.
# Usage: bash scripts/submit_jstars_snaphu_itee_20261119.sh <variant_tag> <EVAL_TS> [array_range]
set -euo pipefail
cd /scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
TAG="$1"; TS="$2"; RANGE="${3:-}"
TREE="data/processed/JSTARS_eval_${TAG}_${TS}"
[[ -d "$TREE" ]] || { echo "missing tree $TREE"; exit 1; }
PAIRLIST="logs/JSTARS_snaphu_pairs_${TAG}_${TS}.txt"
if [[ ! -f "$PAIRLIST" ]]; then
    find "$TREE" -mindepth 2 -maxdepth 2 -type f -name ifg_film_unet.tif -printf '%h\n' \
        | xargs -n1 basename | sort > "$PAIRLIST"
fi
N=$(wc -l < "$PAIRLIST")
[[ "$N" -gt 0 ]] || { echo "no inferred pairs in $TREE"; exit 1; }
RANGE="${RANGE:-1-${N}%48}"
mkdir -p logs/slurm
sbatch --parsable --array="$RANGE" \
    --export=ALL,TREE="$TREE",PAIRLIST="$PAIRLIST" \
    --job-name="JSTARS_snaphu_${TAG}_${TS}" \
    scripts/slurm_jstars_snaphu_itee_20261119.sbatch
