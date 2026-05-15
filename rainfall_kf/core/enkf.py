from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.base import ObservationModel, TransitionModel
from ..utils.validation import ensure_2d_ensemble, ensure_square_matrix, ensure_vector
from .config import EnKFConfig
from .diagnostics import FilterResult
from .ensemble import ensemble_anomalies, ensemble_mean, sample_gaussian


@dataclass(slots=True)
class EnsembleKalmanFilter:
    """Generic Ensemble Kalman Filter that works with black-box models."""

    transition_model: TransitionModel
    observation_model: ObservationModel
    process_noise_cov: np.ndarray
    observation_noise_cov: np.ndarray
    config: EnKFConfig

    def __post_init__(self) -> None:
        self.process_noise_cov = ensure_square_matrix(self.process_noise_cov)
        self.observation_noise_cov = ensure_square_matrix(self.observation_noise_cov)
        self.rng = np.random.default_rng(self.config.random_seed)

    def predict(self, ensemble: np.ndarray, inputs: Any, params: Any) -> np.ndarray:
        """Propagate each ensemble member through the transition model."""

        members = ensure_2d_ensemble(ensemble)
        predicted = np.empty_like(members, dtype=float)

        for index in range(members.shape[1]):
            state_next = self.transition_model(members[:, index], inputs, params, self.rng)
            predicted[:, index] = np.asarray(state_next, dtype=float).reshape(-1)
            predicted[:, index] += sample_gaussian(self.process_noise_cov, self.rng)

        return predicted

    def update(self, predicted_ensemble: np.ndarray, observation: np.ndarray, params: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assimilate an observation and return the analysis ensemble."""

        predicted_ensemble = ensure_2d_ensemble(predicted_ensemble)
        observation = ensure_vector(observation)

        z_ensemble = np.column_stack(
            [
                np.asarray(self.observation_model(predicted_ensemble[:, index], params), dtype=float).reshape(-1)
                for index in range(predicted_ensemble.shape[1])
            ]
        )

        z_mean = ensemble_mean(z_ensemble)
        x_anom = ensemble_anomalies(predicted_ensemble)
        z_anom = ensemble_anomalies(z_ensemble)

        n_members = predicted_ensemble.shape[1]
        p_xz = (x_anom @ z_anom.T) / (n_members - 1)
        p_zz = (z_anom @ z_anom.T) / (n_members - 1) + self.observation_noise_cov
        kalman_gain = p_xz @ np.linalg.pinv(p_zz)

        if self.config.stochastic_update:
            analysis = np.empty_like(predicted_ensemble, dtype=float)
            for index in range(n_members):
                perturbed_obs = observation + sample_gaussian(self.observation_noise_cov, self.rng)
                analysis[:, index] = predicted_ensemble[:, index] + kalman_gain @ (perturbed_obs - z_ensemble[:, index])
        else:
            analysis = predicted_ensemble + kalman_gain @ (observation[:, None] - z_ensemble)

        innovation = observation - z_mean
        whitened_innovation = np.linalg.solve(np.linalg.cholesky(p_zz), innovation)

        return analysis, z_mean, innovation, whitened_innovation

    def step(self, ensemble: np.ndarray, inputs: Any, observation: np.ndarray, params: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one forecast-analysis cycle."""

        predicted = self.predict(ensemble, inputs, params)
        return self.update(predicted, observation, params)

    def run(
        self,
        initial_ensemble: np.ndarray,
        inputs: list[Any] | np.ndarray,
        observations: list[np.ndarray] | np.ndarray,
        params: Any,
    ) -> FilterResult:
        """Run the filter over a time series of inputs and observations."""

        ensemble = ensure_2d_ensemble(initial_ensemble)
        diagnostics = FilterResult()

        for step_index in range(len(observations)):
            ensemble, predicted_observation, innovation, whitened_innovation = self.step(
                ensemble,
                inputs[step_index],
                observations[step_index],
                params,
            )
            diagnostics.append(ensemble, predicted_observation, innovation, whitened_innovation)

        return diagnostics