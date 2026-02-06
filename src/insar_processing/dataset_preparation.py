"""
Dataset preparation utilities for learning-assisted InSAR DEM enhancement.

This module defines light-weight helpers to:
- Tile interferograms, coherence maps, and DEMs into patches,
- Align inputs and targets,
- Build PyTorch-style datasets later on.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from .io import load_raster


@dataclass
class TileConfig:
    tile_size: int = 256
    stride: int = 256


def sliding_window(
    array: np.ndarray, tile_size: int, stride: int
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Generate tiles from a 2D array using a sliding window.

    Yields
    ------
    (row, col, tile) : Tuple[int, int, np.ndarray]
        The upper-left (row, col) index of the tile and the tile array itself.
    """
    h, w = array.shape
    for r in range(0, h - tile_size + 1, stride):
        for c in range(0, w - tile_size + 1, stride):
            yield r, c, array[r : r + tile_size, c : c + tile_size]


def prepare_dem_tiles(
    interferogram_path: str,
    coherence_path: str,
    reference_dem_path: str,
    tile_config: TileConfig,
):
    """
    High-level convenience function to create aligned tiles for ML.

    This is intentionally simple at this stage and can be extended with:
    - Normalization,
    - Metadata features,
    - Train/val/test partitioning.
    """
    igram, _, _ = load_raster(interferogram_path)
    coh, _, _ = load_raster(coherence_path)
    ref_dem, _, _ = load_raster(reference_dem_path)

    assert igram.shape == coh.shape == ref_dem.shape, "Rasters must be co-registered."

    tiles = []
    for r, c, _ in sliding_window(ref_dem, tile_config.tile_size, tile_config.stride):
        ig_tile = igram[r : r + tile_config.tile_size, c : c + tile_config.tile_size]
        coh_tile = coh[r : r + tile_config.tile_size, c : c + tile_config.tile_size]
        dem_tile = ref_dem[r : r + tile_config.tile_size, c : c + tile_config.tile_size]

        tiles.append(
            {
                "row": r,
                "col": c,
                "interferogram": ig_tile,
                "coherence": coh_tile,
                "dem": dem_tile,
            }
        )

    return tiles

