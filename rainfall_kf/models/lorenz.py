from dataclasses import dataclass
import numpy as np

@dataclass()
class LorenzParameters():
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

def LorenzTransition(states: np.ndarray, lorenz_params: LorenzParameters, dt: float = 0.01) -> np.ndarray:
    """
    Transition function for the Lorenz system.
    Vectorized to operate on an ensemble of states.
    Parameters:
    - states: A 2D array of shape (3, EnsembleSize) representing the ensemble of states.
    - lorenz_params: An object containing the Lorenz system parameters (sigma, rho, beta).
    - dt: The time step for the transition.
    Returns:
    - A 2D array of shape (3, EnsembleSize) representing the updated states
    """
    par = lorenz_params
    dxdt = par.sigma * (states[1] - states[0])
    dydt = states[0] * (par.rho - states[2]) - states[1]
    dzdt = states[0] * states[1] - par.beta * states[2]
    return states + np.array([dxdt, dydt, dzdt]) * dt

def LorenzObservation(states: np.ndarray) -> np.ndarray:
    """
    Observation function for the Lorenz system.
    Vectorized to operate on an ensemble of states, observing only the x component.
    Parameters:
    - states: A 2D array of shape (3, EnsembleSize) representing the ensemble of states.
    Returns:
    - A 2D array of shape (1, EnsembleSize) representing the observed values (x component).
    """
    return states[0:1, :]