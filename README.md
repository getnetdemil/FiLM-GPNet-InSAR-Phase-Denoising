Learning-Assisted InSAR DEM Enhancement
======================================

This repository contains a modular research codebase for the paper:

*“Learning-assisted InSAR DEM Enhancement for High-Resolution, Terrain-Aware Hydrologic Digital Twins”*

The goal is to build a learning-assisted InSAR framework that improves interferometric phase stability and DEM quality for hydrologic Digital Twin applications.

## Project structure

- **data/**
  - **raw/**: Original SAR acquisitions and ancillary data.
  - **processed/**: Interferograms, coherence maps, unwrapped phase, intermediate products.
  - **reference/**: High-resolution reference DEMs (e.g., Finnish NLS DEMs).
  - **metadata/**: Acquisition parameters, baselines, orbit info, AOI definitions.

- **src/**
  - **insar_processing/**: Traditional InSAR pipeline components and data preparation.
  - **models/**: Deep learning architectures for coherence enhancement and DEM refinement.
  - **evaluation/**: DEM and interferogram quality metrics, hydrologic-relevant checks.
  - **visualization/**: Plotting utilities and figure-generation scripts.

- **experiments/**
  - **baseline/**: Scripts and configs for traditional InSAR baseline runs.
  - **enhanced/**: Scripts and configs for learning-assisted experiments.

- **notebooks/**: Exploratory analysis, qualitative inspection, and small demos.

- **configs/**
  - **data/**: Sensor- and dataset-specific configuration files (e.g., Sentinel-1, ICEYE).
  - **experiment/**: End-to-end experiment configurations.
  - **model/**: Model architecture hyperparameters.
  - **train/**: Training schedules and optimization settings.

## Getting started

1. **Create and activate an environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. **Organize your data**

   - Place SAR products and interferograms under `data/raw/`.
   - Store derived interferograms, coherence maps, and unwrapped phase under `data/processed/`.
   - Drop high-resolution reference DEMs under `data/reference/`.
   - Keep acquisition geometry, baselines, and AOI information in `data/metadata/`.

3. **Run a small baseline experiment**

   - Configure paths in `configs/data/sentinel1.yaml` and `configs/experiment/baseline_s1.yaml`.
   - Execute a baseline script from `experiments/baseline/` (to be implemented).

4. **Develop learning-assisted methods**

   - Implement or extend models in `src/models/`.
   - Use data preparation utilities in `src/insar_processing/` to create training datasets.
   - Train models via scripts in `experiments/enhanced/`.

## Notes

- This codebase is structured to separate **data processing**, **models**, **evaluation**, and **visualization**, enabling clean experiments and reproducibility.
- The initial implementation should focus on Sentinel-1, with ICEYE and TerraSAR-X/TanDEM-X support added via configuration.

