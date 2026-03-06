"""
Classical interferogram filters for phase noise reduction.

Implementations
---------------
- goldstein()       Goldstein-Werner spectral filter (Goldstein & Werner 1998)
- boxcar()          Multi-look box-car averaging (coherence + phase smoothing)
- adaptive_goldstein()  Coherence-adaptive Goldstein (strength α from coherence)

All functions operate on complex interferograms (Re + j·Im).
Input / output shapes are always (rows, cols) complex64 or complex128.
"""

import numpy as np
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# Goldstein-Werner spectral filter
# ---------------------------------------------------------------------------

def goldstein(
    ifg: np.ndarray,
    alpha: float = 0.5,
    block_size: int = 32,
    overlap: int = 8,
) -> np.ndarray:
    """
    Goldstein-Werner spectral interferogram filter.

    Each overlapping block is transformed to the 2-D frequency domain,
    the spectrum is weighted by |S|^alpha, then transformed back.
    Blocks are recombined with cosine-tapered blending.

    Parameters
    ----------
    ifg : np.ndarray   complex, shape (rows, cols)
    alpha : float      Filter strength in [0, 1].
                       0 = no filtering, 1 = maximum smoothing.
    block_size : int   FFT block size (power of 2 recommended).
    overlap : int      Overlap between adjacent blocks (reduces blocking artefacts).

    Returns
    -------
    np.ndarray   Filtered complex interferogram, same shape as input.

    Reference
    ---------
    Goldstein R.M. & Werner C.L. (1998). "Radar interferogram filtering for
    geophysical applications." Geophysical Research Letters 25(21):4035-4038.
    """
    if not np.iscomplexobj(ifg):
        raise TypeError("Input must be complex.")

    rows, cols = ifg.shape
    step = block_size - overlap
    out = np.zeros_like(ifg, dtype=np.complex128)
    weight = np.zeros((rows, cols), dtype=np.float64)

    # 2-D cosine taper window for smooth block blending
    taper_1d = np.hanning(block_size)
    taper_2d = np.outer(taper_1d, taper_1d)

    for r0 in range(0, rows - block_size + 1, step):
        for c0 in range(0, cols - block_size + 1, step):
            block = ifg[r0:r0 + block_size, c0:c0 + block_size].copy()

            # 2-D FFT
            spec = np.fft.fft2(block)

            # Smooth spectral magnitude (3×3 uniform filter to avoid noise amplification)
            mag = np.abs(spec)
            mag_smooth = uniform_filter(mag, size=3)

            # Goldstein weighting: |S|^alpha (normalised to avoid DC gain)
            max_mag = mag_smooth.max()
            if max_mag > 0:
                w = (mag_smooth / max_mag) ** alpha
            else:
                w = np.ones_like(mag_smooth)

            # Apply filter in frequency domain
            filtered_spec = spec * w
            filtered_block = np.fft.ifft2(filtered_spec)

            # Accumulate with taper weight
            out[r0:r0 + block_size, c0:c0 + block_size] += filtered_block * taper_2d
            weight[r0:r0 + block_size, c0:c0 + block_size] += taper_2d

    # Normalise by accumulated weight (avoid divide-by-zero at edges)
    mask = weight > 0
    result = np.where(mask, out / np.where(mask, weight, 1.0), ifg)
    return result.astype(ifg.dtype)


def adaptive_goldstein(
    ifg: np.ndarray,
    coherence: np.ndarray,
    alpha_min: float = 0.2,
    alpha_max: float = 0.9,
    block_size: int = 32,
    overlap: int = 8,
) -> np.ndarray:
    """
    Coherence-adaptive Goldstein filter.

    Filter strength α(p) = alpha_max × (1 − coherence(p)):
    - High coherence → weak filtering (preserve signal fidelity)
    - Low coherence  → strong filtering (suppress noise)

    Parameters
    ----------
    ifg : np.ndarray        complex, shape (rows, cols)
    coherence : np.ndarray  float [0,1], same shape — per-pixel coherence estimate
    alpha_min : float       minimum α (applied to highest-coherence pixels)
    alpha_max : float       maximum α (applied to lowest-coherence pixels)
    block_size : int
    overlap : int

    Returns
    -------
    np.ndarray  Filtered complex interferogram.
    """
    rows, cols = ifg.shape
    step = block_size - overlap
    out = np.zeros_like(ifg, dtype=np.complex128)
    weight = np.zeros((rows, cols), dtype=np.float64)

    taper_1d = np.hanning(block_size)
    taper_2d = np.outer(taper_1d, taper_1d)

    for r0 in range(0, rows - block_size + 1, step):
        for c0 in range(0, cols - block_size + 1, step):
            block = ifg[r0:r0 + block_size, c0:c0 + block_size].copy()
            coh_block = coherence[r0:r0 + block_size, c0:c0 + block_size]
            mean_coh = float(np.nanmean(coh_block))

            # Adaptive alpha: low coherence → more filtering
            alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - mean_coh)
            alpha = float(np.clip(alpha, alpha_min, alpha_max))

            spec = np.fft.fft2(block)
            mag = np.abs(spec)
            mag_smooth = uniform_filter(mag, size=3)
            max_mag = mag_smooth.max()
            w = (mag_smooth / max_mag) ** alpha if max_mag > 0 else np.ones_like(mag_smooth)

            filtered_block = np.fft.ifft2(spec * w)
            out[r0:r0 + block_size, c0:c0 + block_size] += filtered_block * taper_2d
            weight[r0:r0 + block_size, c0:c0 + block_size] += taper_2d

    mask = weight > 0
    result = np.where(mask, out / np.where(mask, weight, 1.0), ifg)
    return result.astype(ifg.dtype)


# ---------------------------------------------------------------------------
# Multi-look (box-car) coherence + phase
# ---------------------------------------------------------------------------

def boxcar_coherence(
    ifg: np.ndarray,
    slc_ref: np.ndarray,
    slc_sec: np.ndarray,
    looks_range: int = 5,
    looks_azimuth: int = 5,
) -> np.ndarray:
    """
    Estimate interferometric coherence via multi-look box-car averaging.

        γ = |<slc_ref × conj(slc_sec)>| / sqrt(<|slc_ref|²> × <|slc_sec|²>)

    Parameters
    ----------
    ifg : np.ndarray   complex ifg = slc_ref × conj(slc_sec), shape (rows, cols)
    slc_ref, slc_sec : np.ndarray   complex SLCs, same shape
    looks_range : int   number of looks in range (columns)
    looks_azimuth : int number of looks in azimuth (rows)

    Returns
    -------
    np.ndarray float32 [0, 1], same shape — per-pixel coherence estimate.
    """
    ksize = (looks_azimuth, looks_range)

    # Multi-look numerator: |mean of ifg over window|
    re_avg = uniform_filter(ifg.real.astype(np.float64), size=ksize)
    im_avg = uniform_filter(ifg.imag.astype(np.float64), size=ksize)
    numerator = np.sqrt(re_avg ** 2 + im_avg ** 2)

    # Multi-look denominator: sqrt(mean(|ref|²) × mean(|sec|²))
    ref_pwr = uniform_filter(np.abs(slc_ref).astype(np.float64) ** 2, size=ksize)
    sec_pwr = uniform_filter(np.abs(slc_sec).astype(np.float64) ** 2, size=ksize)
    denominator = np.sqrt(ref_pwr * sec_pwr + 1e-30)

    coherence = np.clip(numerator / denominator, 0.0, 1.0)
    return coherence.astype(np.float32)


def multilook(
    arr: np.ndarray,
    looks_range: int = 5,
    looks_azimuth: int = 5,
) -> np.ndarray:
    """
    Multi-look (box-car average) a real or complex array.
    Output is the same shape as input (full-resolution, windowed average).
    """
    ksize = (looks_azimuth, looks_range)
    if np.iscomplexobj(arr):
        re = uniform_filter(arr.real.astype(np.float64), size=ksize)
        im = uniform_filter(arr.imag.astype(np.float64), size=ksize)
        return (re + 1j * im).astype(arr.dtype)
    return uniform_filter(arr.astype(np.float64), size=ksize).astype(arr.dtype)


# ---------------------------------------------------------------------------
# Wrapped phase utilities
# ---------------------------------------------------------------------------

def wrap(phase: np.ndarray) -> np.ndarray:
    """Wrap real-valued phase to (−π, π]."""
    return (phase + np.pi) % (2 * np.pi) - np.pi


def phase_of(ifg: np.ndarray) -> np.ndarray:
    """Return wrapped phase of a complex interferogram."""
    return np.angle(ifg)


def coherence_from_ifg(
    ifg: np.ndarray,
    window: int = 9,
) -> np.ndarray:
    """
    Estimate coherence from a complex interferogram without separate SLCs.

    Uses the ratio of |smoothed ifg| to smoothed |ifg|:
        γ_proxy = |<ifg>| / <|ifg|>

    This is a proxy (slightly biased) but useful when only ifg is available.
    """
    ksize = (window, window)
    smooth_complex_re = uniform_filter(ifg.real.astype(np.float64), size=ksize)
    smooth_complex_im = uniform_filter(ifg.imag.astype(np.float64), size=ksize)
    smooth_abs = uniform_filter(np.abs(ifg).astype(np.float64), size=ksize)

    numerator = np.sqrt(smooth_complex_re ** 2 + smooth_complex_im ** 2)
    denominator = smooth_abs + 1e-30
    return np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)
