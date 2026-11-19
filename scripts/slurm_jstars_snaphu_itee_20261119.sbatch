#!/bin/bash
# =============================================================================
# scripts/slurm_jstars_snaphu_itee_20261119.sbatch
#
# Target the `itee` partition (6x 160-core Skylake nodes) to bypass the
# QOS=interactive maxjobspu=1 cap. Same semantics as the FAST sbatch:
# one pair per task, 8 CPU, SNAPHU --nproc=8. Output identical up to
# float rounding (verified <= 2.3e-5 rad).
# =============================================================================
#SBATCH --account=oy
#SBATCH --partition=itee
#SBATCH --constraint=cpu_skylake
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
