from __future__ import annotations

import numpy as np


def ensemble_mean(ensemble: np.ndarray) -> np.ndarray:
    """Return the column-wise mean state."""

    return np.mean(np.asarray(ensemble, dtype=float), axis=1)


def ensemble_anomalies(ensemble: np.ndarray) -> np.ndarray:
    """Return anomalies relative to the ensemble mean."""

    array = np.asarray(ensemble, dtype=float)
    return array - ensemble_mean(array)[:, None]


def sample_gaussian(covariance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a single zero-mean Gaussian sample with the given covariance."""

    covariance = np.asarray(covariance, dtype=float)
    factor = np.linalg.cholesky(covariance)
    return factor @ rng.standard_normal(covariance.shape[0])