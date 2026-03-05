# Dataset Documentation — DFC 2026
## Capella Space X-Band SAR Temporal Stacks

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Data Products Explained](#2-data-products-explained)
3. [Accessing the Data](#3-accessing-the-data)
4. [Metadata Fields Reference](#4-metadata-fields-reference)
5. [X-Band SAR Signal Properties](#5-x-band-sar-signal-properties)
6. [How We Use the Data](#6-how-we-use-the-data)
7. [Temporal Pair Construction](#7-temporal-pair-construction)
8. [Dataset Statistics (Planned)](#8-dataset-statistics-planned)
9. [Quality Considerations](#9-quality-considerations)
10. [Benchmark Datasets (LEVIR-CD)](#10-benchmark-datasets-levir-cd)

---

## 1. Dataset Overview

The DFC 2026 dataset is provided by **Capella Space**, a commercial SAR
satellite company operating an X-band constellation.

| Property | Value |
|----------|-------|
| Sensor | Capella Space X-band SAR constellation |
| Wavelength | ~3.1 cm (X-band, 9.65 GHz) |
| Polarization | Single (HH or VV) or dual |
| Total collects | ~1,582 unique acquisitions |
| Possible pairs | 17,000+ |
| Geographic coverage | Global (multiple continents) |
| Contest opened | February 4, 2026 |
| License | Capella Open Data (publicly accessible) |

### What Makes This Dataset Special

1. **Very High Resolution** — Spotlight mode provides ~0.5 m resolution,
   far exceeding Sentinel-1 (10–20 m). Individual vehicles, ships, and
   building details are visible.

2. **Dense Temporal Sampling** — Multiple acquisitions of the same location
   throughout the year enable time-series analysis at sub-meter scale.

3. **Diverse Acquisition Geometry** — Images acquired at varying incidence
   angles, orbit directions, and look directions. This provides multi-view
   geometry but requires careful pair selection for change detection.

4. **Commercial Quality** — No public procurement delays; imagery collected
   for operational purposes with high geometric accuracy.

---

## 2. Data Products Explained

Capella provides four product types. **We primarily use GEO.**

### GEO — Geocoded Amplitude Image ⭐ (Primary)

- **Format**: Float32 GeoTIFF (amplitude, not power)
- **Coordinate system**: WGS84 geographic or UTM projected
- **Pixel values**: Linear SAR backscatter amplitude
- **Use for**: Deep learning, visual analysis, change detection
- **Notes**: Already geocoded — no terrain correction needed. Images from
  the same stack will be co-registered to ~sub-pixel accuracy.

```python
# How to read a GEO file
import rasterio
with rasterio.open("CAPELLA_C02_SP_GEO_HH_20240101T120000_20240101T120001.tif") as src:
    amplitude = src.read(1)  # Single band, float32
    # amplitude values: typically 0 to ~5000 (linear scale)
    # In dB: 20 * log10(amplitude), typically -30 to +10 dB for X-band
```

### GEC — Geocoded Ellipsoid Corrected

- **Format**: Float32 GeoTIFF (amplitude)
- **Difference from GEO**: Applies ellipsoid height model to reduce
  terrain-induced geometric distortion
- **Use for**: Areas with significant topography where layover/foreshortening
  would distort GEO results
- **Note**: Not always available; fall back to GEO when absent

### SLC — Single Look Complex

- **Format**: Complex-valued array (I + jQ components)
- **Contains**: Both amplitude AND phase information
- **Use for**: Interferometric processing (InSAR), coherence analysis,
  phase-based deformation measurement
- **Not needed for** our change detection approach (amplitude only)

```python
# How to read amplitude from SLC
import numpy as np
# SLC values are complex; amplitude = |SLC|
# amplitude = np.abs(slc_array)
```

### CPHD — Complex Phase History Data

- **Format**: Raw complex phase history (pre-image-formation)
- **Use for**: Custom SAR processing, advanced experimental analysis
- **Not needed for** our approach

---

## 3. Accessing the Data

### Method 1: STAC Catalog (Recommended)

```bash
pip install pystac-client
python data/explore_stac.py --output_dir data/stac_results
```

The STAC catalog at `https://stacindex.org/catalogs/capella-space-opendata`
lists all available items with metadata. Use `pystac-client` to filter
by location, date, mode, and polarization.

### Method 2: Direct S3 Download

```bash
pip install boto3
# No authentication required — public bucket
aws s3 ls s3://capella-open-data/items/ --no-sign-request
python data/download_capella_data.py --use_s3 --collect_ids id1,id2
```

### Method 3: Interactive Browser

- Visit: `https://stacindex.org/catalogs/capella-space-opendata`
- Browse images visually on the Felt map
- Download thumbnails first for visual inspection

### File Naming Convention

Capella files follow this naming pattern:

```
CAPELLA_{platform}_{mode}_{product}_{polarization}_{start}_{end}.tif

Example:
CAPELLA_C02_SP_GEO_HH_20240101T120000Z_20240101T120001Z.tif
         │    │   │   │   │                │
         │    │   │   │   start_UTC         end_UTC
         │    │   │   polarization (HH/VV/HV/VH)
         │    │   product type (GEO/GEC/SLC)
         │    mode (SP=Spotlight, SS=Sliding Spotlight, SM=Stripmap)
         platform/satellite (C02, C03, etc.)
```

---

## 4. Metadata Fields Reference

The CSV sidecar / STAC properties contain the following fields:

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| `collect_id` | string | Unique acquisition identifier | Use as primary key |
| `start_time` | datetime UTC | Acquisition start time | Precise to milliseconds |
| `end_time` | datetime UTC | Acquisition end time | |
| `platform` | string | Satellite ID (C02, C03, etc.) | Capella constellation member |
| `mode` | string | Imaging mode | Spotlight / Sliding Spotlight / Stripmap |
| `polarization` | list | Transmit-receive polarization | ['HH'] or ['VV'] or ['HH','VV'] |
| `look_direction` | string | Left or right looking | Affects shadow direction |
| `flight_direction` | string | Ascending or descending orbit | |
| `incidence_angle` | float | Center incidence angle (°) | Typically 20°–60° |
| `grazing_angle` | float | Complement of incidence | = 90° − incidence_angle |
| `latitude` | float | Scene center latitude (°) | WGS84 |
| `longitude` | float | Scene center longitude (°) | WGS84 |
| `height` | float | Reference ellipsoid height (m) | At scene center |

### Critical Fields for Pair Filtering

```python
# Good temporal pair criteria:
temporal_gap_days = (t_sec - t_ref).days
assert 7 <= temporal_gap_days <= 180          # Change window
assert abs(incidence_ref - incidence_sec) < 5.0  # Geometry consistency
assert polarization_ref == polarization_sec      # Same polarization
# Prefer: mode_ref == mode_sec (but not required)
```

---

## 5. X-Band SAR Signal Properties

Understanding SAR physics is important for preprocessing and interpreting
change detection results.

### Backscatter Mechanisms

At X-band (3 cm wavelength), backscatter is dominated by:

| Surface Type | Mechanism | Typical dB Value | Change Detection Notes |
|-------------|-----------|-----------------|----------------------|
| Open water (calm) | Specular reflection | −30 to −20 dB | Very low, good background |
| Open water (rough) | Bragg resonance | −20 to −10 dB | Variable with wind |
| Bare soil | Diffuse scattering | −20 to −5 dB | Sensitive to moisture |
| Vegetation | Volume + ground | −15 to −5 dB | Seasonal variability |
| Urban (flat roof) | Specular/diffuse | −15 to +5 dB | Stable unless construction |
| Urban (dihedral) | Double-bounce | +5 to +15 dB | Very bright, stable |
| Metal structures | Specular/trihedal | +10 to +20 dB | Persistent bright targets |

### Why X-Band for Change Detection

- **High sensitivity to change**: 3 cm wavelength ≈ typical surface roughness of human construction
- **Double-bounce from buildings**: Corner reflectors created by building walls + ground → very bright, distinctive signature
- **Sensitive to vegetation phenology**: Cannot penetrate dense canopy → reflects seasonal changes
- **Water body detection**: Specular reflection gives clean contrast to land

### Speckle Characteristics

SAR speckle follows a multiplicative noise model:
```
I_observed = I_true × n_speckle
```
where `n_speckle` follows a Gamma distribution.

For X-band at Spotlight resolution, speckle correlation length ≈ 0.5–2 pixels.
Our Lee filter (window=3×3) effectively suppresses speckle while preserving
fine-scale features (smaller window than the 5×5 we use for C-band Sentinel-1).

### Geometric Effects to Watch For

1. **Layover**: Mountain slopes leaning toward the sensor appear compressed.
   Can cause false changes between acquisitions with different incidence angles.
   → **Mitigation**: Filter pairs with `|Δincidence| < 5°`

2. **Shadow**: Area behind elevated structures (buildings, mountains) not
   illuminated. Appears as very low backscatter.
   → **Not a change**: Compare with optical data or DEM if available.

3. **Specular shift**: Azimuth displacement of moving targets (vehicles, ships).
   Appears as "ghost" targets at wrong location.
   → Can create false change detections; mitigated by temporal averaging.

---

## 6. How We Use the Data

### Preprocessing Pipeline (per image)

```
Raw GEO GeoTIFF (float32 amplitude)
        │
        ▼
[Step 1] Mask nodata and zero values
        │
        ▼
[Step 2] Convert to dB scale:  dB = 20 × log₁₀(amplitude)
        │                      (factor 20 because GEO stores amplitude, not power)
        ▼
[Step 3] Lee speckle filter (3×3 window for X-band)
        │                   - Reduces multiplicative speckle noise
        │                   - Preserves edges and bright targets
        ▼
[Step 4] Clip to X-band dynamic range: [-30 dB, +10 dB]
        │                              - Values outside this range are artifacts
        ▼
[Step 5] Normalize to [0, 1]:  norm = (dB − dB_min) / (dB_max − dB_min)
        │                       norm = (dB − (−30)) / (10 − (−30)) = (dB + 30) / 40
        ▼
[Step 6] Replace remaining NaN with 0 (background noise floor)
        │
        ▼
Output: Float32 numpy array [H, W], range [0, 1]
```

### Temporal Pair Construction (per pair)

```
T1 image (preprocessed)    T2 image (preprocessed)
         │                           │
         └─────────────┬─────────────┘
                       │
              [Align shapes: crop to min(H,W)]
                       │
              [Coregistration check: NCC > 0.2]
                       │
              [Compute pseudo-label:
               1. |T2 - T1| (amplitude difference)
               2. Gaussian smoothing (σ=1.5)
               3. Adaptive threshold (90th percentile)
               4. Sigmoid normalization → [0,1]]
                       │
              [Sliding window patch extraction:
               256×256 patches, 128-pixel stride]
                       │
              [Filter: valid pixel fraction > 80%]
                       │
Output: N × (patch_t1, patch_t2, pseudo_label) triplets
```

---

## 7. Temporal Pair Construction

### Filtering Logic

```python
def is_valid_pair(ref, sec):
    gap = (sec.datetime - ref.datetime).days

    if not (7 <= gap <= 180):
        return False  # Too short (speckle dominates) or too long (geometry drift)

    if ref.polarization != sec.polarization:
        return False  # Different polarizations not directly comparable

    if abs(ref.incidence - sec.incidence) > 5.0:
        return False  # Foreshortening differences create geometric artifacts

    return True
```

### Temporal Gap and Expected Change

The temporal gap influences what types of changes we expect to detect:

| Gap Range | Expected Change Types |
|-----------|----------------------|
| 7–30 days | Fast events: ship movements, construction progress, flooding |
| 30–90 days | Medium events: crop harvesting, building completion, seasonal vegetation |
| 90–180 days | Slow events: urban expansion, large infrastructure, seasonal ice |

### Pair Sampling Strategy

For a stack with N acquisitions: we have N×(N-1)/2 possible pairs.
We include ALL valid pairs (not just consecutive), which:
- Provides more training data diversity
- Allows the model to learn change at multiple temporal scales
- Prevents the model from "cheating" by using temporal ordering

At inference for storytelling, we use consecutive pairs (T1→T2, T2→T3, etc.)
to produce an interpretable temporal narrative.

---

## 8. Dataset Statistics (Planned)

*To be updated once data is downloaded.*

| Metric | Target | Actual |
|--------|--------|--------|
| Temporal stacks selected | 10+ | — |
| Scene types covered | 4+ | — |
| Total image pairs | 200+ | — |
| Total 256×256 patches | 20,000+ | — |
| Training patches | ~14,000 | — |
| Validation patches | ~3,000 | — |
| Test patches | ~3,000 | — |
| Mean temporal gap (days) | — | — |
| Mean change fraction | — | — |
| % patches with >10% change | — | — |

---

## 9. Quality Considerations

### Coregistration

Capella GEO products are geocoded to a common grid, but sub-pixel
misregistration can occur between acquisitions due to:
- Atmospheric propagation delays (up to ~1 pixel)
- Orbital geometry differences
- Interpolation artifacts from different incidence angles

**Our check**: Normalized Cross-Correlation (NCC) in the center region.
- NCC > 0.5: Good coregistration
- NCC 0.2–0.5: Acceptable (warn)
- NCC < 0.2: Poor, skip pair

### Radiometric Consistency

Even "unchanged" areas show backscatter variation between acquisitions due to:
- Wind-induced roughness changes on water surfaces
- Soil moisture changes (rain events)
- Vegetation dielectric changes
- Different antenna look angles

**Implication**: Our pseudo-labels will have noise from these effects.
The soft (continuous) pseudo-labels handle this gracefully — small
amplitude differences get small pseudo-label values, not binary noise.

### Shadow and Layover

These geometric artifacts create regions where change detection is
unreliable. For production use, a DEM-based shadow/layover mask should
be applied. For the contest, we note these areas in our case study analysis.

---

## 10. Benchmark Datasets (LEVIR-CD)

LEVIR-CD is used for **quantitative architectural validation** since
Capella DFC data has no ground truth labels.

### LEVIR-CD Overview

| Property | Value |
|----------|-------|
| Task | Building change detection |
| Source | Google Earth aerial imagery |
| Resolution | 0.5 m/pixel |
| Image size | 1024 × 1024 |
| Patch size (used) | 256 × 256 |
| Training pairs | 445 |
| Validation pairs | 64 |
| Test pairs | 128 |
| Change class | Building footprint changes |
| Labels | Binary change/no-change |
| Download | https://justchenhao.github.io/LEVIR/ |

### Why LEVIR-CD Is Appropriate for Validation

1. **Same spatial scale**: 0.5 m/pixel ≈ Capella Spotlight resolution
2. **Similar change types**: Building construction/demolition (dominant
   change type in urban Capella stacks)
3. **Labeled**: Provides F1/IoU metrics independent of pseudo-label quality
4. **Widely used**: State-of-the-art F1 scores are well-established (~90%+)

### Known Differences Between LEVIR-CD and Capella DFC

| Aspect | LEVIR-CD | Capella DFC |
|--------|----------|-------------|
| Sensor | Optical (Google Earth) | SAR (X-band) |
| Appearance | RGB color | Grayscale amplitude |
| Speckle | None | Present |
| Shadows | Optical shadow | SAR shadow/layover |
| Scene diversity | Urban only | Urban + agriculture + coast |

We train on LEVIR-CD after converting to grayscale and adding synthetic
speckle noise, partially bridging the domain gap.
