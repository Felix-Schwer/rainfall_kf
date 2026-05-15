from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EnKFConfig:
    """Configuration for the Ensemble Kalman Filter."""

    ensemble_size: int
    inflation_factor: float = 1.0
    stochastic_update: bool = False
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        if self.inflation_factor <= 0:
            raise ValueError("inflation_factor must be positive")