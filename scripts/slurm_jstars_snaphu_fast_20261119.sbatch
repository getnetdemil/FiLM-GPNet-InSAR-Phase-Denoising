#!/bin/bash
# =============================================================================
# scripts/slurm_jstars_snaphu_fast_20261119.sbatch
#
# Faster variant of slurm_jstars_snaphu_20261119.sbatch: targets the lightly
# loaded `interactive` partition with 8 CPUs/task (vs. normal/4). Still one
# pair per task; output identical to the --nproc=4 variant up to float
# rounding (verified <= 2.3e-5 rad in smoke test).
#
# Submit via scripts/submit_jstars_snaphu_fast_20261119.sh.
# =============================================================================
#SBATCH --account=oy
#SBATCH --partition=interactive
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

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} tree=$TREE pair=$PAIR start=$(date -u +%FT%TZ) nproc=${SLURM_CPUS_PER_TASK}"
"${ENV_PREFIX}/bin/python" -u scripts/unwrap_snaphu_jstars_20261119.py \
    --pairs_dir "$TREE" --pair_name "$PAIR" \
    --input_ifg ifg_film_unet.tif --output_name unw_phase_film_unet.tif \
    --workers 1 --nproc "${SLURM_CPUS_PER_TASK}"
echo "end=$(date -u +%FT%TZ)"
