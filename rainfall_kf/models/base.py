from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TransitionModel(Protocol):
    """Black-box transition model for one forecast step."""

    def __call__(self, state: np.ndarray, inputs: Any, params: Any, rng: np.random.Generator | None = None) -> np.ndarray:
        ...


@runtime_checkable
class ObservationModel(Protocol):
    """Black-box observation model mapping state to observation space."""

    def __call__(self, state: np.ndarray, params: Any) -> np.ndarray:
        ...