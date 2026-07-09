# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IEEE GRSS 2026 Data Fusion Contest submission (deadline: April 06, 2026 — submitted). Method: **FiLM-GPNet** — self-supervised InSAR phase denoising via geometry-conditioned Noise2Noise. No ground-truth clean interferograms; all training is self-supervised.

**Dataset**: Capella Space X-band SAR SLCs, public AWS S3 (`s3://capella-open-data/data/`, region `us-west-2`, no auth). STAC: `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json`.

**AOIs used**: AOI_000 Hawaii (primary, 221 collects), AOI_024 Western Australia (secondary), AOI_008 Los Angeles (zero-shot transfer).

**Model output**: denoised complex interferogram (2-band) + log-variance uncertainty (1-band). **Not** a DEM directly — DEM is produced downstream via SBAS inversion.

## Environment

The active conda env lives at `/scratch/gdemil24/hrwsi_s3client/torch-gpu`. Use this prefix for all commands:

```bash
export REPO=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
export ENV=/scratch/gdemil24/hrwsi_s3client/torch-gpu
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
```

**Common run prefix** (use for every Python script):
```bash
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u
```

**Critical**: `rasterio` requires `LD_LIBRARY_PATH=$ENV/lib:...` or it throws GLIBCXX errors. `PYTHONPATH=$REPO` is always required (no editable install). Never use `conda activate` — always use `conda run --prefix $ENV`.

## Pipeline — Step-by-Step

The complete working pipeline is documented in `How_2_run_full_SLC.md` with copy-paste commands. Summary:

| Step | Script | Key Output |
|------|--------|------------|
| 1 | `scripts/download_subset.py --index_only` | `data/manifests/full_index.parquet` |
| 2 | `scripts/download_subset.py` | `data/raw/<AOI_ID>/` (SLC GeoTIFF + `_extended.json`) |
| 3 | `scripts/build_pairs_manifest.py` | `*_full_image.parquet` + `*_triplets_full_image.parquet` |
| 4 | `scripts/preprocess_pairs_full_image.py` | `ifg_*.tif`, `coherence.tif`, `coreg_meta.json` per pair |
| 5 | `scripts/unwrap_snaphu.py --input_ifg ifg_goldstein_complex_real_imag.tif` | `unw_phase.tif` |
| 6 | `experiments/enhanced/train_film_unet.py` | `*_final.pt` checkpoint |
| 7 | `train_film_unet.py --resume ... --loss_closure 0.8 --epochs 10` | fine-tuned checkpoint |
| 8 | `scripts/unwrap_snaphu.py --input_ifg ifg_film_unet.tif` | `unw_phase_film_unet.tif` |
| 9 | `scripts/download_copernicus_dem.py` | `data/reference/copernicus_dem/<aoi>_dem.tif` |
| 10 | `eval/compute_metrics.py` | `metrics_comparison.csv` + figures |
| 11 | `eval/compute_metrics.py` (zero-shot, different `--pairs_dir`) | cross-AOI metrics |

Processed pair output directory: `data/processed/<AOI_TAG>_pairs_full_image/<ref_date>__<sec_date>/`

Each pair dir contains:
- `ifg_raw_complex_real_imag.tif` — raw wrapped interferogram (2-band: real, imag)
- `ifg_goldstein_complex_real_imag.tif` — Goldstein-filtered
- `ifg_film_unet.tif` — FiLM-GPNet denoised (written by inference in `compute_metrics.py`)
- `coherence.tif` — interferometric coherence [0, 1]
- `log_var.tif` — per-pixel log-variance from model
- `unw_phase.tif` — SNAPHU-unwrapped Goldstein phase
- `unw_phase_film_unet.tif` — SNAPHU-unwrapped FiLMUNet phase
- `coreg_meta.json` — coregistration metadata + FiLM conditioning vector

## Source Packages (`src/`)

- **`insar_processing/io.py`** — `load_raster()`, `save_raster()`, `resample_raster()`. All raster I/O goes here.
- **`insar_processing/pair_graph.py`** — `PairGraphConfig`, `build_pair_graph()`, `find_triplets()`. Size-agnostic; used by `build_pairs_manifest.py`.
- **`insar_processing/pair_graph_full_image.py`** — Full-image variant with `CapellaMeta`-aware B_perp and tie-point diagnostics integration.
- **`insar_processing/geometry.py`** — B_perp from orbit state vectors; incidence/graze angle computation.
- **`insar_processing/filters.py`** — Goldstein adaptive phase filter; NL-InSAR baseline.
- **`insar_processing/sublook.py`** — Sub-look splitting of SLC data for Noise2Noise training targets.
- **`insar_processing/baseline.py`** — Legacy `BaselineConfig` + `run_baseline()` / `phase_to_height()`. Kept for reference.
- **`insar_processing/dataset_preparation.py`** — `TileConfig`, `sliding_window()`, `prepare_dem_tiles()`. Legacy tiling path; not used in full-image pipeline.
- **`models/film_unet.py`** — `FiLMUNet`: FiLM-conditioned encoder-decoder. Input: `(B, 2, H, W)` complex interferogram. FiLM vector: `(B, 7)`. Output: `(B, 3, H, W)` (denoised real, denoised imag, log-variance).
- **`models/unet_baseline.py`** — `UNetBaseline`: original non-conditioned U-Net. Superseded by `film_unet.py`.
- **`losses/physics_losses.py`** — N2N loss, triplet closure loss, temporal consistency loss, gradient smoothness loss.
- **`evaluation/closure_metrics.py`** — All 5 contest metrics: M1 triplet closure, M2 unwrap success rate, M3 usable pairs, M4 DEM NMAD, M5 temporal residual.
- **`evaluation/dem_metrics.py`** — `rmse()`, `mae()`, `bias()` with optional boolean mask.
- **`visualization/plots.py`** — Plotting helpers for DEM comparison, closure histograms, error maps.

## FiLM Conditioning Vector

All scripts that read `coreg_meta.json` for the FiLM conditioning vector use these 7 keys:

```python
dt    = float(m.get("dt_days", 30.0))        # temporal baseline (days)
inc   = float(m.get("incidence_angle_deg", 45.0))
graze = 90.0 - inc
bperp = float(m.get("bperp_m", 500.0))       # perpendicular baseline (m)
mode  = 1.0 if str(m.get("mode", "")).upper() == "SL" else 0.0
look  = 1.0 if str(m.get("look_direction", "")).upper() == "RIGHT" else 0.0
snr   = float(m.get("snr_proxy", 0.5))
```

`metadata_dim: 7` in `configs/model/film_unet.yaml` must match. All fields use `.get()` with defaults — backward compatible with both old and new `coreg_meta.json` schemas.

## `coreg_meta.json` Schema

Two schemas exist in `data/processed/`:

**Old schema** (cropped pipeline, `preprocess_pairs.py`):
```json
{"patch_size": 4096, "patch_row_ref": N, "patch_col_ref": N,
 "row_offset_px": N, "col_offset_px": N,
 "dt_days": N, "bperp_m": N, "incidence_angle_deg": N,
 "mode": "SL", "look_direction": "RIGHT", "snr_proxy": N,
 "id_ref": "...", "id_sec": "..."}
```

**New schema** (full-image pipeline, `preprocess_pairs_full_image.py`):
```json
{"dt_days": N, "bperp_m": N, "incidence_angle_deg": N,
 "mode": "SL", "look_direction": "RIGHT", "snr_proxy": N,
 "id_ref": "...", "id_sec": "...",
 "offset_model": {"coeffs_drow": [...], "coeffs_dcol": [...], "order": "quadratic", ...},
 "bbox": {...}, "compatibility": {...}}
```

The new schema omits `patch_*` keys (no cropping) and adds polynomial offset model coefficients + a separate `diagnostics.json` per pair.

## Key Scripts

- **`scripts/preprocess_pairs_full_image.py`** — Full-image Capella SLC coregistration pipeline (V5.8). Two-pass: thumbnail phase cross-correlation seed + 9×9 distributed tie-point NCC grid + quadratic polynomial warp model. `--skip-pass2` runs pass-1 only (faster). Writes `diagnostics.json` separately from `coreg_meta.json`.
- **`scripts/preprocess_pairs.py`** — Legacy cropped pipeline (4096×4096 center patch, single cross-correlation offset). Do not use for new work.
- **`scripts/build_pairs_manifest.py`** — Reads `full_index.parquet`, applies filters (`--aoi`, `--no-require-same-platform`), builds pair graph and triplet list.
- **`scripts/unwrap_snaphu.py`** — SNAPHU phase unwrapping. `--input_ifg` selects which interferogram to unwrap; `--output_name` sets output filename. Add `--workers N` for parallelism (memory-intensive; use ≤2 on 64 GB nodes).
- **`scripts/patch_coreg_meta.py`** — Back-fills missing keys into old-schema `coreg_meta.json` files.
- **`scripts/assess_coreg_quality.py`** — Reads `diagnostics.json` files, produces coregistration quality report.
- **`scripts/sbas_dem.py`** — Weighted multi-baseline DEM inversion using all available `unw_phase*.tif` pairs. FiLMUNet confidence weights (`exp(−σ̂)`) vs coherence weights (Goldstein). Outputs `dem_goldstein.tif`, `dem_filmunet.tif`, comparison figure.
- **`eval/compute_metrics.py`** — Runs FiLMUNet inference + computes all 5 contest metrics. `--skip_snaphu_metrics` for inference + M1/M5 only; `--snaphu_only` for M2/M3/M4 from existing unwrapped files.
- **`eval/zero_shot_transfer.py`** — Cross-AOI zero-shot evaluation.

## Contest Metrics

Implemented in `src/evaluation/closure_metrics.py`:

1. **M1 Triplet closure error** — `median(|wrap(φ_ij + φ_jk − φ_ik)|)`, target: ↓ ≥30%
2. **M2 Unwrap success rate** — ≥90% connected component + closure gate, target: ↑ ≥15 pp
3. **M3 Percent usable pairs** — coherence > 0.35 + unwrap pass + closure gate, target: ↑ ≥25%
4. **M4 DEM NMAD** — `1.4826 × median(|e − median(e)|)` vs Copernicus GLO-30, target: ↓ ≥15%
5. **M5 Temporal residual** — `‖Ax − φ̂‖₂` (unweighted SBAS residual), target: ↓ ≥20%

**M5 note**: Use `weights` only for SBAS inversion; report the unweighted residual `‖phi_stack - A @ x_star‖`.

## Checkpoints

- **Hawaii (AOI_000)**: `experiments/enhanced/checkpoints/film_unet/raw2gold_closure_20260321_1852_final.pt`
- **AOI_024 + zero-shot AOI_008**: `experiments/enhanced/checkpoints/film_unet/aoi024_finetune_closure_20260406_1503/aoi024_finetune_closure_20260406_1503_final.pt`

## Data Manifests

Manifests in `data/manifests/`:
- `full_index.parquet` — 791 SLC collects, 39 AOIs
- `full_index_full_image.parquet` — 6,149 Hawaii pairs
- `AOI024_full_index_full_image.parquet` — 909 pairs
- `AOI008_full_index_full_image.parquet` — 2,818 pairs
- `*_triplets_full_image.parquet` — corresponding triplet lists

SHA-256 checksums are in `REPRODUCIBILITY.md`.
