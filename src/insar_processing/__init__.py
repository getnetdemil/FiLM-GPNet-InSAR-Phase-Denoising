"""
InSAR processing package.

This subpackage contains:

- Low-level I/O utilities for SAR, interferograms, coherence maps, and DEMs.
- Baseline InSAR processing helpers (wrappers around external tools or existing products).
- Dataset preparation utilities for training learning-based models.
"""

from . import io  # noqa: F401
from . import baseline  # noqa: F401
from . import dataset_preparation  # noqa: F401

