from __future__ import annotations

import numpy as np


def ensure_2d_ensemble(ensemble: np.ndarray) -> np.ndarray:
    array = np.asarray(ensemble, dtype=float)
    if array.ndim != 2:
        raise ValueError("ensemble must be a 2D array with shape (state_dim, ensemble_size)")
    return array


def ensure_square_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return array


def ensure_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)