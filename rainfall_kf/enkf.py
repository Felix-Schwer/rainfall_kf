from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class FilterResult:
    """Container for filter history and diagnostics (efficient version)."""
    _ensembles: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _predicted_observations: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _innovations: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _whitened_innovations: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    times: np.ndarray
    observations: np.ndarray

    ensembles: np.ndarray | None = field(default=None, init=False)
    predicted_observations: np.ndarray | None = field(default=None, init=False)
    innovations: np.ndarray | None = field(default=None, init=False)
    whitened_innovations: np.ndarray | None = field(default=None, init=False)

    def append(self, ensemble: np.ndarray, predicted_observation: np.ndarray, innovation: np.ndarray, whitened_innovation: np.ndarray, ) -> None:
        """Store one time step."""

        self._ensembles.append(np.asarray(ensemble, dtype=float))
        self._predicted_observations.append(np.asarray(predicted_observation, dtype=float))
        self._innovations.append(np.asarray(innovation, dtype=float))
        self._whitened_innovations.append(np.asarray(whitened_innovation, dtype=float))

    def finalize(self) -> None:
        """Convert lists into stacked numpy arrays (call once after filtering)."""
        self.ensembles = np.stack(self._ensembles, axis=0)
        self.predicted_observations = np.stack(self._predicted_observations, axis=2).squeeze(axis=1)
        self.innovations = np.stack(self._innovations, axis=2).squeeze(axis=1)
        self.whitened_innovations = np.stack(self._whitened_innovations, axis=2).squeeze(axis=1)

        # free memory of lists
        self._ensembles.clear()
        self._predicted_observations.clear()
        self._innovations.clear()
        self._whitened_innovations.clear()

        self.RMSE = np.mean(self.innovations**2, axis=1)**0.5
        self.NSE = 1 - np.sum(self.innovations**2, axis=1) / np.sum((self.observations - np.mean(self.observations, axis=1, keepdims=True))**2, axis=1)
        self.VolBias = np.sum(self.innovations, axis=1) / np.sum(self.observations, axis=1)

    def plot_ensembles(self) -> plt.Figure:
        """Plot ensemble trajectories over time."""
        if self.ensembles is None:
            raise ValueError("Call finalize() before plotting.")
        
        n_states = self.ensembles.shape[1]
        n_ensemble = self.ensembles.shape[2]
        
        fig, axes = plt.subplots(n_states, 1, figsize = (12, 3*n_states))
        axes = np.atleast_1d(axes)

        for i in range(n_states):
            ax = axes[i]
            for j in range(n_ensemble):
                if j == 0:
                    line, = ax.plot(self.times, self.ensembles[:, i, j], color="grey", linewidth=0.5, alpha=0.5)
                    member_handle = line
                else:
                    ax.plot(self.times, self.ensembles[:, i, j], color="grey", linewidth=0.5, alpha=0.5)            
            ensemble_mean = np.mean(self.ensembles[:, i:i+1, :], axis=2)
            mean_handle, =ax.plot(self.times, ensemble_mean, color="black", linewidth=2)
            ax.legend(handles=[mean_handle, member_handle], labels=["Ensemble Mean", "Ensemble Members"])
        return fig
    
    def plot_observations(self) -> plt.Figure:
        """Plot predicted observations and innovations over time."""
        if self.predicted_observations is None:
            raise ValueError("Call finalize() before plotting.")
        
        n_obs = self.predicted_observations.shape[0]

        fig, axes = plt.subplots(n_obs, 1, figsize=(12, 3*n_obs))
        axes = np.atleast_1d(axes)

        for i in range(n_obs):
            ax = axes[i]
            ax.plot(self.times, self.observations[i], color="blue", label="Measurements")
            ax.plot(self.times, self.predicted_observations[i,:], color="black", label="Predicted Observations (Ens. mean)")
            ax.text(0.99, 0.95, f"RMSE: {self.RMSE[i]:.2f}\nNSE: {self.NSE[i]:.2f}\nVol. Bias: {self.VolBias[i]:.2f}",
                    transform=ax.transAxes, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
            ax.legend(loc='upper left')
        return fig
    
    def plot_innovations(self) -> plt.Figure:
        """Plot innovations over time."""
        if self.innovations is None:
            raise ValueError("Call finalize() before plotting.")
        
        n_obs = self.innovations.shape[0]

        fig, axes = plt.subplots(n_obs, 1, figsize=(12, 3*n_obs))
        axes = np.atleast_1d(axes)

        for i in range(n_obs):
            ax = axes[i]
            ax.plot(self.times, self.innovations[i,:], label="Innovation")
            ax.text(0.99, 0.95, f"Mean: {np.mean(self.innovations[i,:]):.2f}\nVar: {np.var(self.innovations[i,:]):.2f}",
                    transform=ax.transAxes, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
            ax.legend(loc='upper left')
        fig.suptitle("Innovations (δ-Residuals)")
        return fig
    
    def plot_whitened_innovations(self) -> plt.Figure:
        """Plot whitened innovations over time."""
        if self.whitened_innovations is None:
            raise ValueError("Call finalize() before plotting.")
        
        n_obs = self.whitened_innovations.shape[0]

        fig, axes = plt.subplots(n_obs, 1, figsize=(12, 3*n_obs))
        axes = np.atleast_1d(axes)

        for i in range(n_obs):
            ax = axes[i]
            ax.plot(self.times, self.whitened_innovations[i,:], label="Whitened Innovation")
            ax.text(0.99, 0.95, f"Mean: {np.mean(self.whitened_innovations[i,:]):.2f}\nVar: {np.var(self.whitened_innovations[i,:]):.2f}",
                    transform=ax.transAxes, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="gray",alpha=0.9))
            ax.legend(loc='upper left')
        fig.suptitle("Normalized Innovations (ϵ-Residuals)")
        return fig
    
    def plot_autocorrelogram(self, max_lag: int = 10) -> plt.Figure:
        """Plot autocorrelogram of innovations."""
        if self.innovations is None:
            raise ValueError("Call finalize() before plotting.")
        
        n_obs = self.whitened_innovations.shape[0]

        fig, axes = plt.subplots(n_obs, 1, figsize=(12, 3*n_obs))
        axes = np.atleast_1d(axes)

        for i in range(n_obs):
            ax = axes[i]
            innovation_series = self.whitened_innovations[i,:]
            lags = np.arange(-max_lag, max_lag + 1)
            autocorr = [np.corrcoef(innovation_series[:-lag], innovation_series[lag:])[0,1] if lag != 0 else 1.0 for lag in lags]
            ax.stem(lags, autocorr, use_line_collection=True)
            ax.axhline(-2/np.sqrt(len(self.times)), color='gray', linestyle='--', label='Confidence Bounds')
            ax.axhline(2/np.sqrt(len(self.times)), color='gray', linestyle='--')
            ax.legend()
        fig.suptitle("Autocorrelogram of Whitened Innovations")
        return fig

@dataclass()
class EnsembleKalmanFilter:
    """
    A class representing an Ensemble Kalman Filter (EnKF) for data assimilation.
    """
    TransitionEquation: callable
    ObservationEquation: callable
    Q: np.ndarray
    R: np.ndarray
    addGaussInputSig: np.ndarray | None = None
    mulLognormInputSig: np.ndarray | None = None
    seed: int | None = None

    def __post_init__(self):
        if self.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed)

    def predict(self, states: np.ndarray, ens_inputs: np.ndarray) -> np.ndarray:
        """
        Predict the next state of the ensemble using the transition equation and process noise.
        """
        predicted_states = self.TransitionEquation(states, ens_inputs)
        predicted_states += self.rng.multivariate_normal(mean=np.zeros(self.n_states), cov=self.Q, size=self.EnsembleSize).T
        return predicted_states
    
    def update(self, predicted_states: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """
        Update the ensemble states using the observation and the Kalman gain.
        """
        predicted_observations = self.ObservationEquation(predicted_states) #y_i
        simulated_obs_noises = self.rng.multivariate_normal(mean=np.zeros(self.n_obs), cov=self.R, size=self.EnsembleSize).T # v_i

        perturbed_pred_observations = predicted_observations + simulated_obs_noises #z_i

        U = (predicted_states - np.mean(predicted_states, axis=1, keepdims=True))/ np.sqrt(self.EnsembleSize - 1)
        V = (perturbed_pred_observations - np.mean(perturbed_pred_observations, axis=1, keepdims=True))/ np.sqrt(self.EnsembleSize - 1)

        K = U @ np.linalg.pinv(V)

        updated_states = predicted_states + K @ (observation + simulated_obs_noises - predicted_observations) # x_i pred + K @ (y + v_i - y_i)

        innovation = observation - np.mean(predicted_observations, axis=1, keepdims=True)
        whitened_innovation = np.linalg.solve(np.linalg.cholesky(V @ V.T), innovation)

        self.result.append(updated_states, np.mean(predicted_observations, axis=1, keepdims=True), innovation, whitened_innovation)

        return updated_states
    
    def broadcast_input(self, inpt: np.ndarray) -> np.ndarray:
        inpt = np.asarray(inpt, dtype=float)
        if inpt.ndim == 1:
            inpt = inpt.reshape(-1, 1)
            ens_inputs = np.repeat(inpt, self.EnsembleSize, axis=1)
        if inpt.shape[1] == 1 and self.EnsembleSize > 1:
            ens_inputs = np.repeat(inpt, self.EnsembleSize, axis=1)
        return ens_inputs
    
    def apply_input_noise(self, ens_inputs: np.ndarray) -> np.ndarray:
        if self.addGaussInputSig is not None:
            ens_inputs += self.rng.normal(loc=0.0, scale=self.addGaussInputSig[:, np.newaxis], size=ens_inputs.shape)
        if self.mulLognormInputSig is not None:
            # Shifting mean of underlying normal distribuation ensures multiplicative noise has mean 1.0
            ens_inputs *= self.rng.lognormal(mean=-(self.mulLognormInputSig[:, np.newaxis]**2)/2, sigma=self.mulLognormInputSig[:, np.newaxis], size=ens_inputs.shape)
        return ens_inputs

    def run(self, initial_ensemble: np.ndarray, times: np.ndarray, inputs: np.ndarray, observations: np.ndarray) -> FilterResult:
        """
        Run the Ensemble Kalman Filter over a sequence of observations.
        """
        states = self.initialize(initial_ensemble, times, inputs, observations)

        self.result = FilterResult(times=times, observations=observations)

        for i, t in enumerate(times):
            inp = inputs[:, i:i+1]
            ens_inputs = self.broadcast_input(inp)
            ens_inputs = self.apply_input_noise(ens_inputs)
            obs = observations[:, i:i+1]
            predicted_states = self.predict(states, ens_inputs)
            states = self.update(predicted_states, obs)

        self.result.finalize()

        return self.result
    
    def initialize(self, initial_ensemble: np.ndarray, times: np.ndarray, inputs: np.ndarray, observations: np.ndarray) -> np.ndarray:
        self.n_states = initial_ensemble.shape[0]
        self.n_obs = observations.shape[0]
        self.n_inputs = inputs.shape[0]
        self.EnsembleSize = initial_ensemble.shape[1]
        self.n_steps = len(times)
        return initial_ensemble