"""
Sub-look splitting for self-supervised Noise2Noise training.

Splits a complex SAR SLC into two independent sub-look images by selecting
alternating range or azimuth frequency sub-bands. The two sub-looks see the
same scene with independent speckle realisations → suitable as a Noise2Noise
training pair without any clean reference.

Reference: Lehtinen et al. (2018) "Noise2Noise: Learning Image Restoration
without Clean Data"; adapted for complex SAR interferograms.

Usage
-----
from src.insar_processing.sublook import split_sublooks_fft, make_n2n_pair

slc = np.load("slc.npy")                   # complex64, shape (rows, cols)
sub1, sub2 = split_sublooks_fft(slc, axis=1, n_looks=2)

# For interferogram Noise2Noise pairs:
ifg_noisy, ifg_target = make_n2n_pair(slc_ref, slc_sec, axis=1)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core sub-look splitting
# ---------------------------------------------------------------------------

def split_sublooks_fft(
    slc: np.ndarray,
    axis: int = 1,
    n_looks: int = 2,
    overlap: float = 0.0,
) -> list[np.ndarray]:
    """
    Split a complex SLC into n_looks independent sub-looks via FFT sub-banding.

    Each sub-look spans 1/n_looks of the full bandwidth along the chosen axis.
    Sub-looks have independent speckle realisations, making them suitable as
    Noise2Noise training pairs.

    Parameters
    ----------
    slc : np.ndarray complex64 or complex128, shape (rows, cols)
        Input complex SAR SLC in image (pixel) domain.
    axis : int
        Frequency axis to split: 0 = azimuth, 1 = range.
    n_looks : int
        Number of sub-looks (typically 2 for N2N pairs).
    overlap : float
        Fractional overlap between adjacent sub-bands [0, 0.5).
        0.0 = no overlap (independent speckle); small values reduce resolution loss.

    Returns
    -------
    list[np.ndarray]
        n_looks complex arrays each of the same shape as slc.
        Each sub-look is back-projected to the full image grid (upsampled
        from sub-band) so they are spatially registered with each other.
    """
    if slc.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {slc.shape}")
    if not np.iscomplexobj(slc):
        raise ValueError("Input must be complex (complex SLC).")

    n = slc.shape[axis]

    # Transform to frequency domain along chosen axis
    spec = np.fft.fft(slc, axis=axis)
    spec = np.fft.fftshift(spec, axes=axis)  # centre DC

    band_width = int(n / n_looks * (1.0 + overlap))
    band_width = min(band_width, n)

    sublooks = []
    for k in range(n_looks):
        centre = int(n / n_looks * (k + 0.5))  # band centre index
        half = band_width // 2
        lo = max(centre - half, 0)
        hi = min(centre + half, n)

        # Zero out all frequencies outside this band
        sub_spec = np.zeros_like(spec)
        if axis == 1:
            sub_spec[:, lo:hi] = spec[:, lo:hi]
        else:
            sub_spec[lo:hi, :] = spec[lo:hi, :]

        # Back to image domain — keep same shape as input
        sub_spec = np.fft.ifftshift(sub_spec, axes=axis)
        sub_img = np.fft.ifft(sub_spec, axis=axis)
        # Scale to preserve amplitude statistics
        sub_img = sub_img * (n / band_width) ** 0.5
        sublooks.append(sub_img.astype(slc.dtype))

    return sublooks


def split_sublooks_odd_even(slc: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Split SLC into two sub-looks by selecting odd/even lines or samples.

    Simpler than FFT sub-banding. Produces a 2× downsampled sub-look pair
    with independent speckle (adjacent lines are decorrelated in SAR).
    Each output has half the rows (axis=0) or columns (axis=1) of the input.

    Parameters
    ----------
    slc : np.ndarray   complex, shape (rows, cols)
    axis : int         0 = split azimuth (rows), 1 = split range (cols)

    Returns
    -------
    sub1, sub2 : np.ndarray   Each is half the size along the split axis.
    """
    if axis == 0:
        return slc[0::2, :], slc[1::2, :]
    return slc[:, 0::2], slc[:, 1::2]


# ---------------------------------------------------------------------------
# Interferogram N2N pair generation
# ---------------------------------------------------------------------------

def make_n2n_pair(
    slc_ref: np.ndarray,
    slc_sec: np.ndarray,
    axis: int = 1,
    method: str = "fft",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Noise2Noise training pair from two registered complex SLCs.

    Strategy
    --------
    1. Split both SLCs into sub-looks 1 and 2.
    2. Form interferogram_A = sub_ref_1 × conj(sub_sec_1)
       Form interferogram_B = sub_ref_2 × conj(sub_sec_2)
    3. Return (ifg_A, ifg_B) — same scene, independent speckle.

    A model trained with input=ifg_A, target=ifg_B learns to denoise
    without ever seeing a clean reference interferogram.

    Parameters
    ----------
    slc_ref, slc_sec : np.ndarray   complex, registered SLC pair, same shape
    axis : int   frequency axis for sub-look splitting (1 = range, typically)
    method : str  "fft" (spectral sub-banding) or "odd_even" (line interleaving)

    Returns
    -------
    ifg_a, ifg_b : np.ndarray complex  N2N input/target interferogram pair
    """
    if slc_ref.shape != slc_sec.shape:
        raise ValueError(
            f"SLCs must have matching shapes: {slc_ref.shape} vs {slc_sec.shape}"
        )

    if method == "fft":
        ref_subs = split_sublooks_fft(slc_ref, axis=axis, n_looks=2)
        sec_subs = split_sublooks_fft(slc_sec, axis=axis, n_looks=2)
    elif method == "odd_even":
        ref_subs = list(split_sublooks_odd_even(slc_ref, axis=axis))
        sec_subs = list(split_sublooks_odd_even(slc_sec, axis=axis))
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'fft' or 'odd_even'.")

    ifg_a = ref_subs[0] * np.conj(sec_subs[0])
    ifg_b = ref_subs[1] * np.conj(sec_subs[1])
    return ifg_a, ifg_b


# ---------------------------------------------------------------------------
# Tile extraction for training
# ---------------------------------------------------------------------------

def extract_sublook_tiles(
    slc_ref: np.ndarray,
    slc_sec: np.ndarray,
    tile_size: int = 256,
    stride: int = 128,
    axis: int = 1,
    method: str = "fft",
    min_coherence: float = 0.2,
) -> list[dict]:
    """
    Slide a window over the SLC pair and return N2N tile dicts.

    Each tile dict contains:
        "ifg_a"   : np.ndarray complex64 (tile_size, tile_size)  — N2N input
        "ifg_b"   : np.ndarray complex64 (tile_size, tile_size)  — N2N target
        "row_off" : int   top-left row offset in original image
        "col_off" : int   top-left col offset in original image

    Tiles with mean coherence below min_coherence are skipped (incoherent regions
    provide no training signal for interferogram denoising).

    Parameters
    ----------
    slc_ref, slc_sec : np.ndarray complex, registered SLCs
    tile_size : int   spatial tile size (same in both dims)
    stride : int      stride between tiles (< tile_size for overlap)
    axis : int        sub-look split axis
    method : str      "fft" or "odd_even"
    min_coherence : float   minimum mean coherence to include a tile
    """
    rows, cols = slc_ref.shape
    tiles = []

    for r in range(0, rows - tile_size + 1, stride):
        for c in range(0, cols - tile_size + 1, stride):
            ref_tile = slc_ref[r:r + tile_size, c:c + tile_size]
            sec_tile = slc_sec[r:r + tile_size, c:c + tile_size]

            ifg_a, ifg_b = make_n2n_pair(ref_tile, sec_tile, axis=axis, method=method)

            # Quick coherence proxy: magnitude of normalised mean ifg
            full_ifg = ref_tile * np.conj(sec_tile)
            coh_proxy = float(
                np.abs(np.mean(full_ifg))
                / (np.mean(np.abs(ref_tile)) * np.mean(np.abs(sec_tile)) + 1e-10)
            )
            if coh_proxy < min_coherence:
                continue

            tiles.append({
                "ifg_a": ifg_a.astype(np.complex64),
                "ifg_b": ifg_b.astype(np.complex64),
                "coherence_proxy": coh_proxy,
                "row_off": r,
                "col_off": c,
            })

    return tiles
