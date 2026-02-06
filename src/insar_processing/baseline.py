"""
Baseline InSAR processing interfaces.

This module is intentionally lightweight at this stage. It assumes that
external tools (e.g., SNAP, ISCE, GAMMA) or separate preprocessing
pipelines have already produced interferograms, coherence maps, and
possibly unwrapped phase products in `data/processed/`.

The goal here is to:
- Load those products,
- Optionally apply simple filtering,
- Convert unwrapped phase to a DEM,
- Provide a clean Python interface for evaluation and ML integration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .io import load_raster, save_raster


@dataclass
class BaselineConfig:
    """Configuration for a single baseline InSAR DEM run."""

    interferogram_path: str
    coherence_path: Optional[str]
    unwrapped_phase_path: str
    output_dem_path: str
    wavelength_m: float
    incidence_angle_deg: float
    perpendicular_baseline_m: float
    # Additional geometry parameters can be added as needed.


def phase_to_height(
    unwrapped_phase: np.ndarray,
    wavelength_m: float,
    incidence_angle_deg: float,
    perpendicular_baseline_m: float,
) -> np.ndarray:
    """
    Convert unwrapped interferometric phase to relative height.

    This is a placeholder using a simplified relationship; in practice
    you would replace this with the full geometry-aware formulation.
    """
    if perpendicular_baseline_m == 0:
        raise ValueError("Perpendicular baseline must be non-zero.")

    k = 4.0 * np.pi / wavelength_m
    theta_rad = np.deg2rad(incidence_angle_deg)

    # Simplified: h ~ phase / (k * sin(theta)) * (lambda * R / B_perp) etc.
    # Here we keep the form symbolic and expect future refinement.
    height = unwrapped_phase / (k * np.sin(theta_rad))
    return height


def run_baseline(config: BaselineConfig) -> Path:
    """
    Run a minimal baseline DEM generation from existing unwrapped phase.

    Parameters
    ----------
    config : BaselineConfig
        Paths and geometry parameters for this run.

    Returns
    -------
    Path
        Path to the output DEM raster.
    """
    uw_phase, transform, meta = load_raster(config.unwrapped_phase_path)

    # Placeholder: identity mask; you can later use coherence to mask low-quality pixels.
    height = phase_to_height(
        unwrapped_phase=uw_phase,
        wavelength_m=config.wavelength_m,
        incidence_angle_deg=config.incidence_angle_deg,
        perpendicular_baseline_m=config.perpendicular_baseline_m,
    )

    out_path = Path(config.output_dem_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save DEM using the same georeferencing as the input unwrapped phase.
    save_raster(str(out_path), height.astype(meta.get("dtype", uw_phase.dtype)), transform, meta)

    return out_path

