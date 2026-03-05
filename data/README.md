# Data Pipeline — DFC 2026

## Overview

This folder contains scripts to discover, download, and organize Capella Space SAR data
from the DFC 2026 dataset.

## Workflow

```
Step 1: explore_stac.py      → Discover available temporal stacks
Step 2: select_stacks.py     → Choose best stacks for change detection
Step 3: download_capella_data.py → Download GEO images
Step 4: preprocessing/      → Preprocess and build patch pairs
```

## Scripts

| Script | Purpose |
|--------|---------|
| `explore_stac.py` | Query Capella STAC catalog, identify temporal stacks, build pair manifest |
| `select_stacks.py` | Score and select diverse stacks by scene type |
| `download_capella_data.py` | Download GEO/SLC/GEC/CPHD products via STAC or S3 |

## Data Access

Capella Open Data is **publicly accessible** — no authentication required.

**Option 1: STAC API (recommended)**
```bash
pip install pystac-client
python explore_stac.py --output_dir stac_results
```

**Option 2: Direct S3**
```bash
pip install boto3
python download_capella_data.py --use_s3 --collect_ids id1,id2 --output_dir ../data/raw
```

**Option 3: QGIS / Browser**
- Browse interactively at: https://stacindex.org/catalogs/capella-space-opendata
- Download GEO TIFFs manually

## Asset Types

| Type | Description | Use For |
|------|-------------|---------|
| **GEO** | Geocoded amplitude GeoTIFF | Deep learning (our primary format) |
| GEC | Geocoded + ellipsoid corrected | Terrain-corrected analysis |
| SLC | Single Look Complex (amplitude+phase) | InSAR processing |
| CPHD | Complex Phase History | Custom processing |
| thumbnail | Quick-look PNG | Visual inspection |

## Expected Directory Structure After Download

```
data/
├── raw/
│   ├── CAPELLA_C02_SP_GEO_HH_20240101/    # One dir per collect
│   │   ├── *_GEO.tif                       # SAR amplitude GeoTIFF
│   │   └── *_metadata.json                 # STAC item metadata
│   └── CAPELLA_C02_SP_GEO_HH_20240215/
│       ├── *_GEO.tif
│       └── *_metadata.json
├── processed/
│   ├── pairs/                              # Temporal pair patch files
│   │   ├── pair_0001_t1.npy
│   │   ├── pair_0001_t2.npy
│   │   └── pair_0001_pseudo_label.npy
│   └── splits/
│       ├── train_pairs.json
│       ├── val_pairs.json
│       └── test_pairs.json
└── stac_results/
    ├── all_acquisitions.csv
    ├── stack_scores.csv
    ├── pair_manifest.json
    └── selected_stacks.json
```

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| min_temporal_gap | 7 days | Minimum gap between T1 and T2 |
| max_temporal_gap | 180 days | Maximum gap (avoids extreme geometry drift) |
| max_incidence_diff | 5° | Max incidence angle difference |
| patch_size | 256×256 pixels | Extraction patch size |
| stride | 128 pixels | Sliding window stride (50% overlap) |

## SAR Characteristics: Capella X-band vs Sentinel-1 C-band

| Property | Capella X-band | Sentinel-1 C-band |
|----------|----------------|-------------------|
| Wavelength | ~3 cm | ~5.6 cm |
| Resolution | 0.5–1m (Spotlight) | 20m (IW) |
| Penetration | Low (surface) | Medium (partial veg) |
| Urban response | Very strong | Strong |
| Vegetation | Low penetration | Partial penetration |
| Dynamic range | High | Moderate |
| Speckle | Fine-grained | Coarser |

**Preprocessing differences from S1:**
- X-band is more sensitive to surface features
- Dynamic range in dB: approximately -30 to +10 dB (vs -25 to 0 for S1)
- Speckle filter window size: 3×3 sufficient (vs 5×5 for S1)
- Normalization: use empirical percentiles from Capella GEO images
