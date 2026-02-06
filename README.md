Learning-Assisted InSAR DEM Enhancement
======================================

This repository contains a modular research codebase for the work:

*“Learning-assisted InSAR DEM Enhancement for High-Resolution, Terrain-Aware Hydrologic Digital Twins”*

The goal is to build a learning-assisted InSAR framework that improves interferometric phase stability and DEM quality for hydrologic Digital Twin applications, while remaining physically consistent with interferometric observables and scalable across multiple SAR sensors.

## Abstract

Accurate Digital Elevation Models (DEMs) at high spatial resolution are a critical prerequisite for terrain-aware hydrologic Digital Twins, where topographic errors directly compromise flow routing, inundation mapping, and hazard prediction. Within the scope of hydrologic digitization, improving the reliability of DEMs derived from spaceborne Interferometric Synthetic Aperture Radar (InSAR) remains a key challenge.

InSAR is a widely used technique for DEM generation, which exploits phase differences and coherence information from two or more SAR acquisitions. While spaceborne InSAR enables large-scale and weather-independent observations, its performance is strongly constrained by sensor geometry, temporal and perpendicular baselines, and surface dynamics. In particular, coherence degradation caused by vegetation cover, soil moisture variability, atmospheric effects, and the presence of wetlands or water bodies leads to noisy interferograms and reduced DEM accuracy in hydrologically relevant environments.

This study investigates a learning-assisted InSAR framework to enhance interferometric data quality and mitigate coherence-related limitations in SAR-derived DEMs. Deep learning based generative and representation-learning models, including diffusion models, generative adversarial networks, and variational autoencoders, are evaluated to support coherence enhancement and artifact suppression in SAR image pairs. The learning components are integrated with established InSAR processing pipelines to improve interferometric phase stability and DEM quality without compromising the physical consistency of the interferometric observables.

Our methodology leverages daily repeat-track ICEYE and bistatic TerraSAR-X/TanDEM-X satellite acquisitions, with high-resolution reference DEMs from the National Land Survey of Finland enabling robust validation beyond open global DEM products. Initial experiments using Sentinel-1 image pairs show consistent improvements in interferogram quality and spatial coherence patterns relative to baseline InSAR processing, particularly in vegetated and mixed land-cover areas affected by decorrelation. Quantitative aspect of the methodology focuses on improvements in interferogram quality, elevation accuracy, and uncertainty patterns relevant for hydrologic Digital Twin applications. The workflow is designed for scalability across different SAR sensor configurations.

By addressing coherence limitations through the integration of physics-aware deep learning with InSAR pipelines, this work aims to enable more reliable, high-resolution DEM generation for terrain-sensitive hydrologic Digital Twins within the Digital Waters (DIWA) framework.

## Repository overview

This codebase is organized to cleanly separate **data**, **processing pipelines**, **models**, **experiments**, **evaluation**, and **visualization**:

- **`data/`** – inputs and reference products
  - **`data/raw/`**: Original SAR acquisitions and ancillary inputs (e.g., Sentinel-1, ICEYE, TerraSAR-X/TanDEM-X scenes, orbit files).
  - **`data/processed/`**: Interferograms, coherence maps, unwrapped phase, and intermediate InSAR products produced by external tools or this codebase.
  - **`data/reference/`**: High-resolution reference DEMs (e.g., Finnish NLS DEMs) used for validation and training targets.
  - **`data/metadata/`**: Acquisition parameters (temporal/perpendicular baselines, incidence angles), sensor configuration, AOI definitions, and any auxiliary GIS layers.

- **`src/`** – main Python package
  - **`src/insar_processing/`** – traditional InSAR processing and data preparation
    - **`__init__.py`**: Initializes the `insar_processing` subpackage.
    - **`io.py`**: Low-level I/O helpers for loading and saving rasters (interferograms, coherence maps, DEMs) with `rasterio`, plus resampling utilities for grid alignment.
    - **`baseline.py`**: Minimal baseline InSAR DEM generation from existing unwrapped phase, using simplified phase-to-height conversion based on acquisition geometry.
    - **`dataset_preparation.py`**: Sliding-window tiling (`TileConfig`, `sliding_window`) and `prepare_dem_tiles` to create aligned input/target tiles for ML (interferogram + coherence → DEM).
  - **`src/models/`** – deep learning models for learning-assisted enhancement
    - **`__init__.py`**: Package initialization.
    - **`unet_baseline.py`**: Compact U-Net-style CNN (`UNetBaseline`) for DEM refinement given stacked interferogram and coherence channels. Serves as a baseline for more advanced models (GANs, VAEs, diffusion).
  - **`src/evaluation/`** – quantitative evaluation utilities
    - **`__init__.py`**: Package initialization.
    - **`dem_metrics.py`**: DEM quality metrics such as RMSE, MAE, and bias, with optional masks for focusing on hydrologically relevant areas.
  - **`src/visualization/`** – plotting and figure generation
    - **`__init__.py`**: Package initialization.
    - **`plots.py`**: Visualization helpers to compare baseline vs enhanced vs reference DEMs, and to plot error histograms for analysis and publications.

- **`experiments/`** – executable scripts that tie configs, data, and code together
  - **`experiments/baseline/`**
    - **`run_baseline.py`**: Command-line entry point for running a baseline InSAR DEM experiment. Reads a YAML configuration, constructs a `BaselineConfig`, calls `run_baseline`, and writes an output DEM raster.
  - **`experiments/enhanced/`**
    - **`train_unet.py`**: Command-line training script for the U-Net baseline model. Loads data/model/train configs, prepares DEM tiles, runs a simple training loop, and saves a checkpoint.

- **`configs/`** – YAML configuration files for data, experiments, and models
  - **`configs/data/`**
    - **`sentinel1_example.yaml`**: Example configuration pointing to a Sentinel-1 scene (interferogram, coherence, reference DEM) and tiling parameters.
  - **`configs/experiment/`**
    - **`baseline_sentinel1.yaml`**: Example baseline InSAR DEM experiment configuration, including unwrapped phase path, output DEM path, and basic geometric parameters (wavelength, incidence, baseline).
  - **`configs/model/`**
    - **`unet_baseline.yaml`**: U-Net hyperparameters such as number of input/output channels and feature widths.
  - **`configs/train/`**
    - **`default.yaml`**: Training hyperparameters (learning rate, number of epochs, checkpoint directory).

- **`notebooks/`**
  - Jupyter notebooks for exploratory analysis, quick visual sanity checks, and figure prototyping (e.g., domain overview, data inspection, baseline vs enhanced DEM comparison).

- **`requirements.txt`**
  - Python dependencies for InSAR processing, geospatial I/O, and deep learning (NumPy, SciPy, rasterio, PyTorch, etc.).

## Installation

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. **Editable install (optional but recommended)**

   From the repository root:

   ```bash
   pip install -e .
   ```

   This lets you import the package as `import src` or configure a proper package name later.

## Data organization

1. **Raw SAR data and ancillary information**
   - Place sensor-specific SAR acquisitions under subfolders of `data/raw/`, for example:
     - `data/raw/sentinel1/...`
     - `data/raw/iceye/...`
     - `data/raw/tdx/...` (TerraSAR-X/TanDEM-X)
   - Store orbit files, DEMs used during external processing, and any masks in the same hierarchy or in `data/metadata/`.

2. **Processed interferometric products**
   - Place interferograms, coherence maps, and unwrapped phase outputs in `data/processed/`, organized by sensor or experiment:
     - `data/processed/sentinel1/example_interferogram.tif`
     - `data/processed/sentinel1/example_coherence.tif`
     - `data/processed/sentinel1/example_unwrapped_phase.tif`

3. **Reference DEMs**
   - Place high-resolution reference DEMs (e.g., Finnish NLS DEM) in `data/reference/`, ideally mirroring the same AOIs as your interferometric products:
     - `data/reference/sentinel1/example_reference_dem.tif`

4. **Metadata**
   - Store acquisition geometry (incidence angle, perpendicular baseline), temporal baselines, and AOI definitions in structured formats (CSV, JSON, or GeoJSON) under `data/metadata/`.

## Baseline InSAR DEM workflow

The baseline workflow assumes you already have an unwrapped interferometric phase product for a given AOI.

1. **Configure the baseline experiment**

   Edit `configs/experiment/baseline_sentinel1.yaml`:

   - Set `unwrapped_phase_path` to your unwrapped phase raster.
   - Optionally set `coherence_path` if you plan to use coherence for masking later.
   - Set `output_dem_path` to where the baseline DEM should be written.
   - Update `wavelength_m`, `incidence_angle_deg`, and `perpendicular_baseline_m` with sensor-specific parameters.

2. **Run the baseline script**

   ```bash
   python experiments/baseline/run_baseline.py \
       --config configs/experiment/baseline_sentinel1.yaml
   ```

   This will:

   - Load the unwrapped phase raster.
   - Convert phase to height using a simplified geometry-based model.
   - Save a DEM raster at `output_dem_path`.

3. **Evaluate against a reference DEM**

   In a notebook or small script, you can:

   - Load the baseline DEM and reference DEM with `src.insar_processing.io.load_raster`.
   - Align grids if necessary (e.g., using the resampling helper in `io.py`).
   - Compute RMSE, MAE, and bias with `src.evaluation.dem_metrics`.

## Learning-assisted DEM enhancement workflow

The learning-assisted pipeline uses interferograms and coherence maps to learn corrections to the DEM (or directly generate improved DEMs), in line with the abstract.

1. **Configure data tiling**

   Edit `configs/data/sentinel1_example.yaml`:

   - Set `interferogram_path` to your interferogram raster.
   - Set `coherence_path` to the corresponding coherence map.
   - Set `reference_dem_path` to the high-resolution reference DEM.
   - Adjust `tile_size` and `stride` for your GPU memory and dataset size.

2. **Configure the model and training**

   - `configs/model/unet_baseline.yaml`: choose `in_channels`, `out_channels`, and feature depths for `UNetBaseline`.
   - `configs/train/default.yaml`: set `learning_rate`, `num_epochs`, and `output_dir` for checkpoints.

3. **Run the training script**

   ```bash
   python experiments/enhanced/train_unet.py \
       --data_config configs/data/sentinel1_example.yaml \
       --model_config configs/model/unet_baseline.yaml \
       --train_config configs/train/default.yaml
   ```

   This will:

   - Tile the interferogram, coherence, and reference DEM into patches via `prepare_dem_tiles`.
   - Train a U-Net-style CNN to map input patches to DEM patches.
   - Save the final model weights to `experiments/enhanced/checkpoints/unet_baseline_final.pt` (by default).

4. **Apply the trained model to full scenes (future extension)**

   A future script (e.g., `experiments/enhanced/apply_unet.py`) can:

   - Load the trained model.
   - Slide a window over full interferogram/coherence rasters.
   - Assemble an enhanced DEM raster for evaluation and hydrologic use.

## Visualization and analysis

- Use `src/visualization.plots.plot_dem_comparison` to visually compare:
  - Baseline DEM vs enhanced DEM vs reference DEM.
- Use `src/visualization.plots.plot_error_histogram` to inspect error distributions (prediction − reference) and assess bias and spread.
- Combine these with hydrologic analysis tools (e.g., flow routing from DEMs) to relate DEM improvements to hydrologic Digital Twin performance.

## Roadmap and extensions

- **Model extensions**: Add physics-aware loss terms, conditional GANs, VAEs, and diffusion models as described in the abstract.
- **Sensor generalization**: Introduce additional data configs for ICEYE and TerraSAR-X/TanDEM-X, ensuring consistent tiling and evaluation.
- **Hydrologic validation**: Integrate flow accumulation, channel extraction, and inundation metrics to directly quantify benefits for hydrologic Digital Twins.

