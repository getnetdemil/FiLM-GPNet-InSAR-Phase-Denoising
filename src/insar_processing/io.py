"""
I/O utilities for InSAR products and DEMs.

These functions provide thin wrappers around common geospatial libraries
so that the rest of the codebase does not depend on a specific file format.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling


def load_raster(path: str) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """
    Load a single-band raster (e.g., DEM, coherence) as a NumPy array.

    Returns
    -------
    data : np.ndarray
        2D array of raster values.
    transform : rasterio.Affine
        Affine transform mapping pixel coordinates to geographic coordinates.
    meta : dict
        Raster metadata (CRS, dtype, etc.).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raster not found: {p}")

    with rasterio.open(p) as src:
        data = src.read(1)
        meta = src.meta.copy()
        transform = src.transform

    return data, transform, meta


def save_raster(path: str, data: np.ndarray, transform, meta: dict) -> None:
    """
    Save a single-band raster to disk.

    Parameters
    ----------
    path : str
        Output file path.
    data : np.ndarray
        2D array of raster values.
    transform : rasterio.Affine
        Affine transform for the output.
    meta : dict
        Base metadata (CRS, dtype, etc.). Will be updated with shape and count.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    out_meta = meta.copy()
    out_meta.update(
        {
            "height": int(data.shape[0]),
            "width": int(data.shape[1]),
            "count": 1,
        }
    )

    with rasterio.open(p, "w", **out_meta) as dst:
        dst.write(data, 1)


def resample_raster(
    data: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_width: int,
    dst_height: int,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Resample a raster into a new grid.

    This is useful when aligning interferograms, coherence maps, and DEMs.
    """
    dst = np.empty((dst_height, dst_width), dtype=data.dtype)
    with rasterio.Env():
        rasterio.warp.reproject(
            source=data,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=resampling,
        )
    return dst

