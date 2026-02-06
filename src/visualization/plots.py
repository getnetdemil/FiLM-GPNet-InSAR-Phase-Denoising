"""
Visualization utilities for DEMs, interferograms, and error maps.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_dem_comparison(
    baseline_dem: np.ndarray,
    enhanced_dem: np.ndarray,
    reference_dem: Optional[np.ndarray] = None,
    cmap: str = "terrain",
):
    """
    Side-by-side DEM comparison figure.
    """
    n_cols = 2 if reference_dem is None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes = np.atleast_1d(axes)

    im0 = axes[0].imshow(baseline_dem, cmap=cmap)
    axes[0].set_title("Baseline DEM")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(enhanced_dem, cmap=cmap)
    axes[1].set_title("Enhanced DEM")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if reference_dem is not None and n_cols == 3:
        im2 = axes[2].imshow(reference_dem, cmap=cmap)
        axes[2].set_title("Reference DEM")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_error_histogram(
    errors: np.ndarray,
    bins: int = 50,
    title: str = "DEM Error Histogram",
):
    """
    Plot histogram of DEM errors (prediction - reference).
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(errors.ravel(), bins=bins, alpha=0.8)
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()
    return fig

