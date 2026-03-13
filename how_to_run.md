# How to Run — Learning-Assisted InSAR DEM Enhancement

All commands assume you are in the repo root and use the `torch-gpu` conda env.
Always set the rasterio library path first:

```bash
export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
alias py="conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python"
```

---

## Step 1 — Build the SLC Manifest (one-time)

**Script**: `scripts/download_subset.py`
**Purpose**: Crawl the Capella STAC catalog, assign AOI labels to all 791 SLCs,
and save a metadata manifest to parquet. Does not download any data.

```bash
py scripts/download_subset.py \
    --index_only \
    --out_manifest data/manifests/full_index.parquet
```

Output: `data/manifests/full_index.parquet` — 791 rows, 25+ columns including
`aoi_id`, `collect_id`, `inc_angle_deg`, `orbit_state`, `look_dir`, `start_datetime`.

---

## Step 2 — Download SLCs for an AOI

**Script**: `scripts/download_subset.py`
**Purpose**: Parallel S3 download of raw SLC GeoTIFFs + extended JSON sidecars
for a chosen AOI. No AWS credentials needed (unsigned access).

```bash
# Download all Hawaii (AOI_000) collects — ~497 GB, use 8 workers
py scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_000 \
    --out_dir data/raw/ \
    --n_workers 8 \
    --assets slc,metadata

# Download only spotlight ascending collects, cap at 50:
py scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_000 \
    --mode_filter spotlight \
    --orbit_filter ascending \
    --max_collects 50 \
    --out_dir data/raw/ \
    --n_workers 8
```

Output: `data/raw/AOI_000/<collect_id>/` — one subdirectory per collect containing
the CInt16 SLC GeoTIFF and `*_extended.json` metadata sidecar.

---

## Step 3 — Build the Pair Graph + Compute B_perp

**Module**: `src/insar_processing/pair_graph.py` (used internally by `preprocess_pairs.py`)
**Purpose**: Enumerate all valid interferometric pairs, score by Q_ij, and enumerate
closure triplets. Also computes perpendicular baseline B_perp from satellite state vectors.

Run via the pair-graph builder (no standalone CLI — integrated into preprocess step):

```bash
# To regenerate the pair manifests independently:
py -c "
from insar_processing.pair_graph import build_pair_graph, enumerate_triplets
import pandas as pd
manifest = pd.read_parquet('data/manifests/full_index.parquet')
hawaii = manifest[manifest.aoi_id == 'AOI_000']
pairs_df = build_pair_graph(hawaii, dt_max=365, dinc_max=5.0)
pairs_df.to_parquet('data/manifests/hawaii_pairs.parquet', index=False)
triplets_df = enumerate_triplets(pairs_df, dt_max=60, dinc_max=2.0)
triplets_df.to_parquet('data/manifests/hawaii_triplets_strict.parquet', index=False)
print(f'{len(pairs_df)} pairs, {len(triplets_df)} triplets')
"
```

Outputs:
- `data/manifests/hawaii_pairs.parquet` — 8,834 pairs with `q_score`, `bperp_m`, `dt_days`
- `data/manifests/hawaii_triplets_strict.parquet` — 24,171 strict triplets (Δt≤60d, Δinc≤2°)

---

## Step 4 — Preprocess Pairs (Coregistration → Interferogram → Coherence → Goldstein)

**Script**: `scripts/preprocess_pairs.py`
**Purpose**: Full end-to-end preprocessing for each pair: load SLC patches, sub-pixel
coregistration via phase cross-correlation, form complex interferogram, estimate coherence
with a 5×5 box-car window, apply Goldstein-Werner spectral filter. Writes one output
directory per pair. Skips already-processed pairs.

```bash
# Process top-100 pairs by Q_score (used for training):
py scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 100 \
    --workers 4

# Process with coherence-adaptive Goldstein (stronger filter for low-coherence areas):
py scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 100 \
    --adaptive \
    --workers 4

# Single pair for debugging:
py scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 1

# All options:
#   --dt_max 60.0          Max temporal baseline filter (days, default 60)
#   --dinc_max 2.0         Max incidence angle difference filter (deg, default 2)
#   --patch_size 4096      SLC patch size in pixels (default 4096)
#   --looks_range 5        Range looks for coherence window (default 5)
#   --looks_azimuth 5      Azimuth looks for coherence window (default 5)
#   --goldstein_alpha 0.5  Goldstein filter strength α ∈ [0,1] (default 0.5)
#   --adaptive             Use coherence-adaptive α instead of fixed
#   --workers 4            Parallel workers
```

Output per pair in `data/processed/pairs/<pair_id>/`:
- `ifg_raw.tif` — 2-band float32 GeoTIFF (Re, Im of unnormalised interferogram)
- `ifg_goldstein.tif` — 2-band float32 GeoTIFF (Goldstein-filtered Re, Im)
- `coherence.tif` — 1-band float32 GeoTIFF, values ∈ [0,1]
- `coreg_meta.json` — dict with keys `dt_days`, `bperp_m`, `inc_angle_deg`,
  `graze_angle_deg`, `orbit_state`, `look_dir`, `q_score`, `row_offset`, `col_offset`

---

## Step 5 — Phase Unwrapping with SNAPHU

**Script**: `scripts/unwrap_snaphu.py`
**Purpose**: For each preprocessed pair, run SNAPHU to convert the wrapped phase
(-π to π) into the absolute unwrapped phase. Uses coherence for pixel weighting.
Skips already-unwrapped pairs.

**Prerequisite**: install SNAPHU first:
```bash
conda install --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu -c conda-forge snaphu
```

```bash
# Unwrap all 100 preprocessed pairs (DEFO mode for deformation, 2 workers):
py scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --max_pairs 100 \
    --mode DEFO \
    --workers 2

# Use TOPO mode for DEM estimation:
py scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --mode TOPO \
    --workers 2

# All options:
#   --pairs_dir            Dir containing per-pair subdirs (required)
#   --out_dir              Output dir (default: same as pairs_dir)
#   --max_pairs            Cap number of pairs to process
#   --mode DEFO|TOPO       SNAPHU cost function mode (default DEFO)
#   --coh_threshold 0.1    Pixels below this coherence are masked as NaN (default 0.1)
#   --workers 2            Parallel workers
#   --snaphu_bin snaphu    Path to SNAPHU binary (default: 'snaphu' from PATH)
```

Output per pair: `unw_phase.tif` — float32 GeoTIFF of unwrapped phase in radians,
NaN at masked (incoherent) pixels. Required for Metrics 2, 3, 4, 5.

---

## Step 6 — Train FiLMUNet

**Script**: `experiments/enhanced/train_film_unet.py`
**Purpose**: Self-supervised training of the geometry-conditioned FiLM U-Net.
Reads processed pair tiles, splits into train/val/test by temporal order,
trains for 50 epochs with the combined physics loss suite (N2N + uncertainty NLL
+ closure + temporal + gradient). Checkpoints every 5 epochs.

```bash
# Full training run (fresh start):
py experiments/enhanced/train_film_unet.py \
    --data_config configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml

# Resume from a checkpoint:
py experiments/enhanced/train_film_unet.py \
    --data_config configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --resume experiments/enhanced/checkpoints/epoch_020.pt
```

Checkpoints written to `experiments/enhanced/checkpoints/`:
- `epoch_005.pt`, `epoch_010.pt`, ... — periodic saves
- `best_closure.pt` — best validation closure loss

Key config knobs (edit the YAML files):
- `configs/data/capella_aoi_selection.yaml` — `processed_dir`, `max_pairs`, `min_coherence`, `tile_size`, `stride`
- `configs/model/film_unet.yaml` — `features`, `meta_dim`, `embed_dim`
- `configs/train/contest.yaml` — `lr`, `epochs`, `batch_size`, `weight_decay`, loss weights

---

## Step 7 — Run Baseline DEM Experiment (legacy)

**Script**: `experiments/baseline/run_baseline.py`
**Purpose**: Simplified phase-to-height conversion using the classical formula
(no ML). Useful as the baseline to beat for all 5 contest metrics.

```bash
py experiments/baseline/run_baseline.py \
    --config configs/experiment/baseline_sentinel1.yaml
```

---

## Step 8 — Compute Contest Metrics

**Module**: `src/evaluation/closure_metrics.py`
**Purpose**: Compute all 5 IEEE GRSS contest evaluation metrics over a directory
of processed (and optionally unwrapped) pairs.

No standalone CLI yet — run via Python:

```bash
py -c "
from evaluation.closure_metrics import compute_baseline_metrics
results = compute_baseline_metrics(
    pairs_dir='data/processed/pairs',
    triplets_manifest='data/manifests/hawaii_triplets_strict.parquet',
)
import json; print(json.dumps(results, indent=2))
"
```

Metrics returned:
1. `triplet_closure_error` — median/mean/std/rmse of |wrap(φ_ij+φ_jk−φ_ik)|
2. `unwrap_success_rate` — fraction of γ≥0.35 pixels with valid unwrap
3. `usable_pairs_fraction` — fraction passing dual coherence+closure gate
4. `dem_nmad` — 1.4826 × median(|err − median(err)|)
5. `temporal_consistency_residual` — ‖W(Ax*−φ̂)‖₂ from SBAS inversion

---

## GitHub Issue Workflow

```bash
# Create all project issues (idempotent — safe to re-run):
bash scripts/create_github_issues.sh

# List open issues:
gh issue list --repo getnetdemil/Learning-Assisted-InSAR-DEM-Enhancement

# View a specific issue:
gh issue view 17

# Close an issue manually:
gh issue close 17 --comment "Done — training complete."
```

---

## Typical Full Pipeline (end-to-end)

```bash
export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# 1. Build manifest (already done — skip if full_index.parquet exists)
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \
    python scripts/download_subset.py --index_only \
    --out_manifest data/manifests/full_index.parquet

# 2. Download Hawaii SLCs (already done — 497 GB in data/raw/AOI_000/)
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \
    python scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_000 --out_dir data/raw/ --n_workers 8

# 3. Preprocess top-100 pairs (already done — 100 dirs in data/processed/pairs/)
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \
    python scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 100 --workers 4

# 4. Install SNAPHU then unwrap (PENDING — SNAPHU not yet installed)
# conda install --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu -c conda-forge snaphu
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \
    python scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --max_pairs 100 --mode DEFO --workers 2

# 5. Train FiLMUNet (IN PROGRESS — epoch 21/50 as of 2026-03-13)
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \
    python experiments/enhanced/train_film_unet.py \
    --data_config configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --resume experiments/enhanced/checkpoints/epoch_020.pt

# 6. Compute contest metrics
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \
    python -c "
from evaluation.closure_metrics import compute_baseline_metrics
import json
print(json.dumps(compute_baseline_metrics('data/processed/pairs',
    'data/manifests/hawaii_triplets_strict.parquet'), indent=2))
"
```
