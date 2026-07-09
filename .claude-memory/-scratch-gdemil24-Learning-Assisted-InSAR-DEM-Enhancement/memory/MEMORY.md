# Project Memory — Learning-Assisted InSAR DEM Enhancement

## Environment
- Conda env: `torch-gpu` at `/scratch/gdemil24/hrwsi_s3client/torch-gpu`
- Run commands: `conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python ...`
- PyTorch 2.4.0, CUDA=True, boto3, pystac, rasterio all confirmed working

## Data Access
- S3: `capella-open-data`, region `us-west-2`, no auth (`Config(signature_version=UNSIGNED)`)
- STAC: `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json`
- 791 SLC items (filter `_SLC_` in href), 791 GEO (ignored)

## Manifests (already built)
- `data/manifests/full_index.parquet` — 791 SLC rows, 39 AOIs assigned
- `data/manifests/hawaii_pairs.parquet` — 8,834 pairs with B_perp
- `data/manifests/hawaii_triplets_strict.parquet` — 24,171 triplets

## AOI Decision
- **Primary**: AOI_000 (Hawaii) — 221 collects, both orbits, inc 35.8–56.3°, volcanic terrain
- **Secondary**: AOI_008 (LA) for zero-shot transfer demo; AOI_024 (W. Australia) for stable baseline

## Processed Pairs Status — 224 dirs
- `data/processed/pairs/` — 224 dirs total (162 unique valid pairs from coreg_meta.json)
- unw_phase.tif: 224/224 ✓ | unw_phase_film_unet.tif: 224/224 ✓
- Complete triplets: 62

## FINAL Metric Results — authoritative table in memory file
- [Authoritative Metrics Table](feedback_metrics_reference.md) — all values across AOI000/AOI024/AOI008 (confirmed 2026-04-07)

## FINAL Metric Results — see authoritative table in memory
→ See [Authoritative Metrics Table](feedback_metrics_reference.md) for all values across AOI000/AOI024/AOI008.
Do NOT use inline numbers here — always check that file first.

Key checkpoints:
- Hawaii: `raw2gold_closure_20260321_1852_final.pt`
- AOI024 + zero-shot AOI008: `aoi024_finetune_closure_20260406_1503_final.pt`
- M3 Usable Pairs: 0.000 for Hawaii (all pairs fail M1 < 0.5 gate — NOT reported in paper)
- Reference DEM: `data/reference/copernicus_dem/hawaii_dem.tif` (14400×14400, 9 GLO-30 tiles)

## Paper State (2026-04-01)
File: `Latex_Paper_Temporal_SAR_Change/main.tex`
- Table 1: M1/M2/M4/M5 all filled with real numbers, captioned as "our protocol" (not contest metrics)
- Conclusion: explicitly maps to all 9 contest evaluation criteria (soundness, originality, insightfulness, usefulness, temporal data use, scalability, clarity, visuals, reproducibility)
- 7D conditioning vector confirmed throughout: `[Δt, θ_inc, θ_graze, B_perp, mode, look, SNR]`
- Ablation table: V1 (N2N only) vs V4 (full) — footnote cleaned up (no "pending" language)
- User handles push to Overleaf git (`git push origin master` from paper dir)

## Contest Evaluation Criteria Mapping
Contest grades on 9 qualitative dimensions (NOT a numerical leaderboard):
- Soundness → physics-grounded losses (closure, SBAS, sub-look independence)
- Originality → FiLM conditioning + N2N for geometry-diverse SAR stacks
- Insightfulness → M5 (−68.3%) >> M1 (−10.1%): temporal > pairwise sensitivity
- Usefulness → uncertainty-weighted SBAS, no extra supervision needed
- Effective temporal data use → 127-epoch SBAS network, M5 aggregates whole stack
- Scalability → 791-SLC dataset in <15 min on single GPU
- Reproducibility → GitHub + STAC endpoint + SHA-256 checksums

## Critical Runtime Notes
- **SNAPHU**: always run with direct Python path: `/scratch/gdemil24/hrwsi_s3client/torch-gpu/bin/python -u scripts/unwrap_snaphu.py`
- **eval/compute_metrics.py**: must set `PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement`
- **rasterio GLIBCXX error**: always set `export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH`
- **preprocess_pairs.py**: always use `PYTHONPATH=...` (no editable install)

## Key Design Decisions
- Model output: denoised complex interferogram (2ch) + log-variance uncertainty (1ch), NOT DEM directly
- FiLM conditioning: 7D `[Δt, θ_inc, θ_graze, B_perp, mode, look, SNR]` (metadata_dim: 7 in config)
- Training: raw→Goldstein supervised denoising (N2N proxy) + closure + temporal losses
- SBAS weights: `W(p) = 1/σ²(p)` from FiLMUNet; **report unweighted residual for M5**
- M5 formula: use `weights` only for SBAS inversion; report `‖phi_stack - A @ x_star‖` (unweighted)

## Commit Style
- Never include `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` in commit messages.
- Never include "Generated with Claude Code" in PR descriptions, commit messages, or any file.

## AOI Assignment Bug Fix
- DO NOT use groupby with tuple keys — use `drop_duplicates().itertuples()`:
  ```python
  unique_grids = sorted(df[["grid_lon","grid_lat"]].drop_duplicates().itertuples(index=False, name=None))
  aoi_map = {grid: f"AOI_{idx:03d}" for idx, grid in enumerate(unique_grids)}
  ```
