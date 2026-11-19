#!/bin/bash
# =============================================================================
# scripts/slurm_jstars_snaphu_multi_20261119.sbatch
#
# Multi-partition SNAPHU: SLURM picks whichever of {itee, normal} has a slot
# free first. The normal partition's Milan nodes (cn1-6) have hundreds of GB
# free RAM today, and our 2-h / 8-CPU / 4.8-GB jobs are small enough to
# backfill into slots that big pending normal jobs can't fill yet.
# =============================================================================
#SBATCH --account=oy
#SBATCH --partition=itee,normal
#SBATCH --constraint=cpu_milan|cpu_skylake
#SBATCH --cpus-per-task=8
#SBATCH --mem=4800
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/JSTARS_snaphu_%x_%A_%a.log

set -euo pipefail
REPO=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
ENV_PREFIX=/scratch/gdemil24/hrwsi_s3client/torch-gpu
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}"
cd "$REPO"

: "${TREE:?TREE not set}" ; : "${PAIRLIST:?PAIRLIST not set}"
PAIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PAIRLIST")
[[ -n "$PAIR" ]] || { echo "no pair at line ${SLURM_ARRAY_TASK_ID}"; exit 1; }

# Skip if output already exists (race safety when multi-partition races itee)
if [[ -f "${TREE}/${PAIR}/unw_phase_film_unet.tif" ]]; then
    echo "already unwrapped: $PAIR"; exit 0
fi

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} tree=$TREE pair=$PAIR start=$(date -u +%FT%TZ) nproc=${SLURM_CPUS_PER_TASK}"
"${ENV_PREFIX}/bin/python" -u scripts/unwrap_snaphu_jstars_20261119.py \
    --pairs_dir "$TREE" --pair_name "$PAIR" \
    --input_ifg ifg_film_unet.tif --output_name unw_phase_film_unet.tif \
    --workers 1 --nproc "${SLURM_CPUS_PER_TASK}"
echo "end=$(date -u +%FT%TZ)"
