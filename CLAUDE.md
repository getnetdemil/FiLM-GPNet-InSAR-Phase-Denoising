# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is an **active contest submission** for the **IEEE GRSS 2026 Data Fusion Contest** (deadline: April 06, 2026). The `src/` modules contain working skeleton implementations that must be significantly extended. See `plan.md` for the full contest-aligned development plan.

**Critical shift from original design**: The dataset is Capella Space X-band SAR (NOT Sentinel-1), accessed via public AWS S3 (`s3://capella-open-data/data/`, region `us-west-2`). The ML approach is **self-supervised** (Noise2Noise + physics-consistency losses) — there are no ground-truth clean interferograms. The model output is a **denoised complex interferogram + per-pixel uncertainty**, not a DEM directly.

## Environment Setup

```bash
conda create -n insar-dem python=3.10
conda activate insar-dem
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .  # editable install to use `src` as a package
# Contest-specific additional packages
pip install boto3 pystac capella-reader dask geopandas
conda install -c conda-forge isce3  # for coregistration and orbit geometry
```

Verify setup:
```python
import torch; print(torch.cuda.is_available())
import rasterio
import pystac, boto3  # contest data access
```

Validate S3 data access (no authentication required):
```bash
aws s3 ls --no-sign-request s3://capella-open-data/data/
```

## Running Experiments

**Baseline InSAR DEM (phase-to-height conversion):**
```bash
python experiments/baseline/run_baseline.py --config configs/experiment/baseline_sentinel1.yaml
```

**U-Net training:**
```bash
python experiments/enhanced/train_unet.py \
    --data_config configs/data/sentinel1_example.yaml \
    --model_config configs/model/unet_baseline.yaml \
    --train_config configs/train/default.yaml
```

No test suite exists yet. Model checkpoint saves to `experiments/enhanced/checkpoints/unet_baseline_final.pt`.

## Architecture

### Data Flow

```
data/raw/          (SAR SLC acquisitions - not in git)
    |
    v (external: SNAP / ISCE2 / GAMMA)
data/processed/    (interferograms, coherence maps, unwrapped phase .tif)
    |
    v (src/insar_processing/)
data/reference/    (high-res reference DEMs for training targets)
    |
    v (dataset_preparation.py: sliding-window tiling)
    tiles (interferogram + coherence -> target DEM patches)
    |
    v (experiments/enhanced/train_unet.py)
    model checkpoint (.pt)
```

### Source Packages (`src/`)

- **`insar_processing/io.py`** — `load_raster()`, `save_raster()`, `resample_raster()` using rasterio. All raster I/O goes through here.
- **`insar_processing/baseline.py`** — `BaselineConfig` dataclass + `run_baseline()` which calls `phase_to_height()` (simplified geometry: `h = phase / (k * sin(theta))`). Note: the perpendicular baseline parameter is stored in config but not currently used in the height formula.
- **`insar_processing/dataset_preparation.py`** — `TileConfig`, `sliding_window()` generator, `prepare_dem_tiles()` which requires all input rasters to be pre-aligned (same shape).
- **`models/unet_baseline.py`** — `UNetBaseline`: encoder-decoder with skip connections. Input: `(B, 2, H, W)` (interferogram + coherence stacked). Output: `(B, 1, H, W)` (DEM).
- **`evaluation/dem_metrics.py`** — `rmse()`, `mae()`, `bias()` with optional boolean mask.
- **`visualization/plots.py`** — Plotting helpers for DEM comparison and error histograms.

### Config Structure (`configs/`)

YAML configs are loaded with `yaml.safe_load()` (no Hydra at runtime yet, though `hydra-core` is in requirements):
- `configs/data/` — paths to interferogram, coherence, reference DEM; tile_size, stride
- `configs/experiment/` — paths + sensor geometry (wavelength_m, incidence_angle_deg, perpendicular_baseline_m)
- `configs/model/` — in_channels, out_channels, features list
- `configs/train/` — learning_rate, num_epochs, output_dir

### Key Design Decisions

- All rasters must be **co-registered before tiling** — `prepare_dem_tiles()` asserts equal shapes.
- The current training loop in `train_unet.py` processes tiles **one-by-one without DataLoader batching** (scaffold only; replace with proper `Dataset`/`DataLoader`).
- Input tensor format: `(B, 2, H, W)` where channel 0 = interferogram, channel 1 = coherence.
- Model outputs relative height (no absolute reference); absolute DEM requires an external reference elevation.

### Contest-Required Components (not yet implemented)

These are the critical missing pieces for the contest:

| File | Purpose |
|------|---------|
| `src/insar_processing/pair_graph.py` | Pair-graph construction, edge Q_ij scoring |
| `src/insar_processing/geometry.py` | B_perp from ISCE3 orbit geometry |
| `src/insar_processing/filters.py` | Goldstein, NL-InSAR, BM3D baselines |
| `src/insar_processing/sublook.py` | Sub-look splitting for Noise2Noise training |
| `src/models/film_unet.py` | FiLM-conditioned U-Net (replaces `unet_baseline.py`) |
| `src/losses/physics_losses.py` | N2N, closure-consistency, temporal, gradient losses |
| `src/evaluation/closure_metrics.py` | All 5 contest metrics |
| `scripts/download_subset.py` | STAC crawl → manifest → S3 download |
| `scripts/preprocess_pairs.py` | Coregistration → interferogram → coherence |
| `scripts/unwrap_snaphu.py` | SNAPHU unwrapping with coherence mask |
| `eval/compute_metrics.py` | Contest metrics → tables + paper figures |
| `REPRODUCIBILITY.md` | Contest-required: STAC URL, checksums, seeds |

### Contest Success Metrics

All implemented in `src/evaluation/closure_metrics.py`:

1. **Triplet closure error** — `wrap(φ_ij + φ_jk − φ_ik)`, target: median ↓ ≥30%
2. **Unwrap success rate** — ≥90% connected component coverage + closure gate, target: ↑ ≥15 pp
3. **Percent usable pairs** — coherence > 0.35 + unwrap pass + closure gate, target: ↑ ≥25%
4. **DEM NMAD** — `1.4826 × median(|e − median(e)|)`, target: ↓ ≥15%
5. **Temporal consistency residual** — `‖W(Ax − φ̂)‖₂`, target: ↓ ≥20%

### Key Design Decisions (Contest-Driven)

- **Self-supervised training**: Noise2Noise via sub-look splits — no reference clean data needed
- **Model output**: denoised complex interferogram (2ch) + log-variance uncertainty (1ch), NOT a DEM
- **FiLM conditioning**: model is conditioned on `[Δt, Δθ_inc, Δθ_graze, B_perp, mode, look, SNR_proxy]`
- **AOI-based train/val/test splits**: prevents geographic leakage (not tile-based random splits)
- **Uncertainty integration**: predicted `σ²(p)` weights SNAPHU and SBAS inversion downstream
