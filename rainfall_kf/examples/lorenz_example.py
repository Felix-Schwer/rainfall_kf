from __future__ import annotations

"""Lorenz-63 example for validating the generic Ensemble Kalman Filter.

This example is intentionally small and self-contained so it can be used as a
sanity check in a notebook or Colab runtime before wiring in HBV.
"""

from dataclasses import dataclass

import numpy as np

from ..core.config import EnKFConfig
from ..core.enkf import EnsembleKalmanFilter


@dataclass(slots=True)
class Lorenz63Params:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01


def lorenz63_rhs(state: np.ndarray, params: Lorenz63Params) -> np.ndarray:
    x, y, z = np.asarray(state, dtype=float).reshape(3)
    dx = params.sigma * (y - x)
    dy = x * (params.rho - z) - y
    dz = x * y - params.beta * z
    return np.array([dx, dy, dz], dtype=float)


def lorenz63_transition(
    state: np.ndarray,
    inputs: None,
    params: Lorenz63Params,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Advance the Lorenz-63 system one step with RK4."""

    del inputs, rng
    state = np.asarray(state, dtype=float).reshape(3)
    dt = params.dt

    k1 = lorenz63_rhs(state, params)
    k2 = lorenz63_rhs(state + 0.5 * dt * k1, params)
    k3 = lorenz63_rhs(state + 0.5 * dt * k2, params)
    k4 = lorenz63_rhs(state + dt * k3, params)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def lorenz63_observation(state: np.ndarray, params: Lorenz63Params) -> np.ndarray:
    """Observe the x component only."""

    del params
    return np.asarray(state, dtype=float).reshape(3)[:1]


def simulate_truth(initial_state: np.ndarray, n_steps: int, params: Lorenz63Params) -> np.ndarray:
    truth = np.zeros((3, n_steps + 1), dtype=float)
    truth[:, 0] = np.asarray(initial_state, dtype=float).reshape(3)

    state = truth[:, 0]
    for step in range(1, n_steps + 1):
        state = lorenz63_transition(state, None, params)
        truth[:, step] = state

    return truth


def build_observations(truth: np.ndarray, observation_std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, observation_std, size=(1, truth.shape[1]))
    return truth[:1, :] + noise


def build_initial_ensemble(
    initial_state: np.ndarray,
    ensemble_size: int,
    spread: float,
    rng: np.random.Generator,
) -> np.ndarray:
    initial_state = np.asarray(initial_state, dtype=float).reshape(3, 1)
    perturbations = rng.normal(0.0, spread, size=(3, ensemble_size))
    return initial_state + perturbations


def run_demo(
    n_steps: int = 2000,
    ensemble_size: int = 50,
    observation_std: float = 2.0,
    process_std: float = 0.5,
    seed: int = 1,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Run a complete Lorenz-63 EnKF demonstration."""

    rng = np.random.default_rng(seed)
    params = Lorenz63Params()

    truth = simulate_truth(np.array([1.0, 1.0, 1.0]), n_steps, params)
    observations = build_observations(truth, observation_std, rng)

    config = EnKFConfig(
        ensemble_size=ensemble_size,
        stochastic_update=False,
        random_seed=seed,
    )
    enkf = EnsembleKalmanFilter(
        transition_model=lorenz63_transition,
        observation_model=lorenz63_observation,
        process_noise_cov=np.eye(3) * (process_std ** 2),
        observation_noise_cov=np.eye(1) * (observation_std ** 2),
        config=config,
    )

    initial_ensemble = build_initial_ensemble(truth[:, 0], ensemble_size, spread=1.5, rng=rng)
    inputs = [None] * n_steps
    observations_list = [observations[:, step] for step in range(1, n_steps + 1)]

    result = enkf.run(initial_ensemble, inputs, observations_list, params)

    analysis_means = np.column_stack([ensemble.mean(axis=1) for ensemble in result.ensembles])
    truth_aligned = truth[:, 1 : n_steps + 1]

    return {
        "truth": truth_aligned,
        "observations": observations[:, 1 : n_steps + 1],
        "analysis_means": analysis_means,
        "result": result,
    }


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()