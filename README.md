Learning-Assisted InSAR DEM Enhancement
========================================

Contest submission for the **[IEEE GRSS 2026 Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2026-ieee-grss-data-fusion-contest/)**
— *deadline: April 06, 2026.*

This repository implements a **stack-aware, self-supervised deep learning pipeline** for improving Interferometric SAR (InSAR) products over large, geometry-diverse satellite stacks, with downstream improvements in DEM quality and time-series consistency.

## Overview

The contest provides a Capella Space X-band SAR dataset: ~1,582 unique collects enabling 17,000+ interferometric pairs across multiple AOIs, with substantial diversity in acquisition mode, incidence angle, look direction, and orbital geometry.

A competitive entry must be **pair-graph-aware** and **geometry-conditioned**. Our approach:

1. **Pair-graph construction** — build a graph (nodes = collects, edges = candidate pairs), score each edge by a quality formula `Q_ij` using temporal baseline, incidence difference, perpendicular baseline, and SNR proxies, then select a high-value subset for processing.
2. **Baseline InSAR products** — coregistration (ISCE3 + capella-reader), interferogram formation, coherence estimation, classical filtering (Goldstein, NL-InSAR, BM3D), SNAPHU unwrapping.
3. **Self-supervised DL enhancement** — a FiLM-conditioned U-Net trained without any ground-truth labels using Noise2Noise (sub-look splits) + closure-consistency + temporal-consistency + fringe-preservation losses. Outputs a denoised complex interferogram and per-pixel uncertainty map.
4. **Uncertainty-weighted inversion** — predicted uncertainty weights SNAPHU and SBAS stack inversion to improve unwrapping and time-series products.
5. **Evaluation** — five hard, physics-linked contest metrics computed against classical baselines.

## Contest Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Triplet closure error | `median|wrap(φ_ij + φ_jk − φ_ik)|` on stable pixels | ↓ ≥30% |
| Unwrap success rate | % interferograms with ≥90% connected coverage + closure gate | ↑ ≥15 pp |
| Percent usable pairs | % edges passing coherence + unwrap + closure gates | ↑ ≥25% |
| DEM NMAD | `1.4826 × median(|e − median(e)|)`, stable terrain | ↓ ≥15% |
| Temporal consistency residual | SBAS inversion residual `‖W(Ax − φ̂)‖₂` | ↓ ≥20% |

## Installation

```bash
conda create -n insar-dem python=3.10
conda activate insar-dem

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install -r requirements.txt
pip install -e .

# Contest-specific packages
pip install boto3 pystac capella-reader dask geopandas
conda install -c conda-forge isce3
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
aws s3 ls --no-sign-request s3://capella-open-data/data/
```

## Data Access

Data is on a **public AWS S3 bucket** — no authentication required.

- **S3 bucket**: `s3://capella-open-data/data/` (region: `us-west-2`)
- **STAC catalog**: `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json`
- **Contest collection**: child link `"IEEE Data Contest 2026"` → `capella-open-data-ieee-data-contest/collection.json`

The catalog is a **static STAC** (JSON files, no `/search` endpoint) — use `pystac` directly.

```bash
# Crawl catalog, select AOIs, build local manifest, download subset
python scripts/download_subset.py \
    --stac_root https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json \
    --out_manifest data/manifests/subset_manifest.csv \
    --out_dir data/raw/
```

Data format per collect: COG GeoTIFF (CInt16 complex SLC) + STAC JSON + extended JSON sidecar.

## Workflows

### 1. Baseline InSAR Pipeline

```bash
# Coregistration, interferogram formation, coherence, classical filtering, SNAPHU
python scripts/preprocess_pairs.py \
    --manifest data/manifests/subset_manifest.csv \
    --out_dir data/processed/

python scripts/unwrap_snaphu.py \
    --processed_dir data/processed/ \
    --coherence_threshold 0.35

# Simplified phase-to-height DEM (existing scaffold)
python experiments/baseline/run_baseline.py \
    --config configs/experiment/baseline_sentinel1.yaml
```

### 2. Self-Supervised DL Training

```bash
python experiments/enhanced/train_film_unet.py \
    --data_config configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml
```

The model (`FiLMUNet`) takes a complex interferogram (2 channels: Re, Im) conditioned on acquisition geometry metadata, and outputs a denoised interferogram + per-pixel log-variance. Training is fully self-supervised via sub-look splits — no clean reference interferograms are required.

### 3. Evaluation

```bash
python eval/compute_metrics.py \
    --processed_dir data/processed/ \
    --ref_dem data/reference/ \
    --out_dir results/
```

Produces all five contest metrics per AOI and aggregated, plus figures for the paper.

## Repository Structure

```
src/
  insar_processing/
    io.py                  # Rasterio-based raster I/O (load_raster, save_raster, resample_raster)
    baseline.py            # BaselineConfig + phase-to-height conversion (run_baseline)
    dataset_preparation.py # Sliding-window tiling (TileConfig, sliding_window, prepare_dem_tiles)
    pair_graph.py          # [to build] pair-graph construction + Q_ij edge scoring
    geometry.py            # [to build] B_perp from ISCE3 orbit geometry
    filters.py             # [to build] Goldstein, NL-InSAR, BM3D baselines
    sublook.py             # [to build] sub-look splits for Noise2Noise training
  models/
    unet_baseline.py       # Basic U-Net (in_channels=2 → out_channels=1)
    film_unet.py           # [to build] FiLM-conditioned U-Net for contest submission
  losses/
    physics_losses.py      # [to build] N2N, closure, temporal, gradient losses
  evaluation/
    dem_metrics.py         # RMSE, MAE, bias
    closure_metrics.py     # [to build] all 5 contest metrics
  visualization/
    plots.py               # DEM comparison + error histogram plots

experiments/
  baseline/run_baseline.py              # Phase-to-height DEM (existing scaffold)
  enhanced/train_unet.py               # Basic U-Net training (existing scaffold)
  enhanced/train_film_unet.py          # [to build] Contest DL training script

scripts/                               # [to build] Download, preprocess, unwrap
eval/                                  # [to build] Contest metrics + figures
configs/
  data/          # Data paths, AOI selection, tiling parameters
  experiment/    # InSAR geometry parameters
  model/         # Model architecture hyperparameters
  train/         # Learning rate, epochs, checkpoint directory
notebooks/       # Exploratory analysis and pair-graph visualization
```

## Development Plan

See [`plan.md`](plan.md) for the week-by-week implementation schedule, detailed phase descriptions, metric definitions, DL loss formulas, ablation study design, and the reproducibility checklist required for contest submission.

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (to be created) for the STAC root URL, contest collection ID, exact download manifest with checksums, fixed random seeds, and deterministic training settings — all required by the contest.
