"""
DEM quality metrics for evaluating baseline and learning-assisted methods.
"""

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    diff = pred - target
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    diff = np.abs(pred - target)
    return float(np.mean(diff))


def bias(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    diff = pred - target
    return float(np.mean(diff))

