#!/bin/bash
# =============================================================================
# scripts/slurm_jstars_snaphu_20261119.sbatch
#
# SLURM array: SNAPHU-unwrap one JSTARS eval pair per task.
#   ifg_film_unet.tif -> unw_phase_film_unet.tif   inside a JSTARS_eval_* tree.
# Pair list file: one pair directory name per line; array index = line number (1-based).
#
# Submit (one array per variant):
#   sbatch --array=1-224%48 \
#     --export=ALL,TREE=data/processed/JSTARS_eval_<tag>_<TS>,PAIRLIST=logs/JSTARS_snaphu_pairs_<tag>_<TS>.txt \
#     --job-name=JSTARS_snaphu_<tag> scripts/slurm_jstars_snaphu_20261119.sbatch
# =============================================================================
#SBATCH --account=oy
#SBATCH --partition=normal
#SBATCH --constraint=cpu_milan|cpu_skylake
#SBATCH --cpus-per-task=4
#SBATCH --mem=4800
#SBATCH --time=03:00:00
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

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} tree=$TREE pair=$PAIR start=$(date -u +%FT%TZ)"
"${ENV_PREFIX}/bin/python" -u scripts/unwrap_snaphu_jstars_20261119.py \
    --pairs_dir "$TREE" --pair_name "$PAIR" \
    --input_ifg ifg_film_unet.tif --output_name unw_phase_film_unet.tif \
    --workers 1 --nproc "${SLURM_CPUS_PER_TASK}"
echo "end=$(date -u +%FT%TZ)"
