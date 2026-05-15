from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class FilterResult:
    """Container for filter history and diagnostics."""

    ensembles: list[np.ndarray] = field(default_factory=list)
    predicted_observations: list[np.ndarray] = field(default_factory=list)
    innovations: list[np.ndarray] = field(default_factory=list)
    whitened_innovations: list[np.ndarray] = field(default_factory=list)

    def append(
        self,
        ensemble: np.ndarray,
        predicted_observation: np.ndarray,
        innovation: np.ndarray,
        whitened_innovation: np.ndarray,
    ) -> None:
        self.ensembles.append(np.asarray(ensemble, dtype=float))
        self.predicted_observations.append(np.asarray(predicted_observation, dtype=float))
        self.innovations.append(np.asarray(innovation, dtype=float))
        self.whitened_innovations.append(np.asarray(whitened_innovation, dtype=float))