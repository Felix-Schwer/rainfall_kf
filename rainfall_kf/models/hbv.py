from dataclasses import dataclass

import numpy as np


@dataclass()
class HBVParameters():
    d: float = 0.0
    fc: float = 0.0
    beta: float = 0.0
    c: float = 0.0
    k0: float = 0.0
    l: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    kp: float = 0.0
    pwp: float = 0.0
    tt: float = 0.0


def _as_parameter_vector(params: HBVParameters | np.ndarray) -> np.ndarray:
    if isinstance(params, HBVParameters):
        return np.array([
            params.d,
            params.fc,
            params.beta,
            params.c,
            params.k0,
            params.l,
            params.k1,
            params.k2,
            params.kp,
            params.pwp,
            params.tt,
        ], dtype=float).reshape(-1, 1)

    params_array = np.asarray(params, dtype=float)
    if params_array.ndim == 1:
        return params_array.reshape(-1, 1)
    return params_array


def _broadcast_inputs(inputs: np.ndarray, ensemble_size: int) -> np.ndarray:
    inputs_array = np.asarray(inputs, dtype=float)
    if inputs_array.ndim == 1:
        inputs_array = inputs_array.reshape(-1, 1)
    if inputs_array.shape[1] == 1 and ensemble_size > 1:
        inputs_array = np.repeat(inputs_array, ensemble_size, axis=1)
    return inputs_array


def HBVTransition(states: np.ndarray, inputs: np.ndarray, params: HBVParameters | np.ndarray) -> np.ndarray:
    """
    Transition function for the HBV rainfall-runoff model.

    The state and input arrays follow the same column-vector convention as the
    MATLAB code, but vectorized across ensemble members.
    """
    states_array = np.asarray(states, dtype=float)
    if states_array.ndim == 1:
        states_array = states_array.reshape(-1, 1)

    inputs_array = _broadcast_inputs(inputs, states_array.shape[1])
    params_array = _as_parameter_vector(params).reshape(-1)

    snow = states_array[0:1, :]
    soil = states_array[1:2, :]
    s1 = states_array[2:3, :]
    s2 = states_array[3:4, :]

    temp = inputs_array[0:1, :]
    prec = inputs_array[1:2, :]
    temp_m = inputs_array[2:3, :]
    dpem = inputs_array[3:4, :]

    d, fc, beta, c, k0, l, k1, k2, kp, pwp, tt = params_array

    below_threshold = temp < tt
    melt = d * (temp - tt)

    snow_next = np.where(below_threshold, snow + prec, np.maximum(snow - melt, 0.0))
    lwater = np.where(below_threshold, 0.0, prec + np.minimum(snow, melt))

    pe = (1.0 + c * (temp - temp_m)) * dpem
    ea = np.where(soil > pwp, pe, pe * (soil / pwp))
    dq = lwater * ((soil / fc) ** beta)

    s1_next = s1 + dq - (np.maximum(0.0, s1 - l) * k0) - (s1 * k1) - (s1 * kp)
    s2_next = s2 + (s1 * kp) - (s2 * k2)
    soil_next = soil + lwater - dq - ea

    return np.vstack([snow_next, soil_next, s1_next, s2_next, s1])


def HBVObservation(states: np.ndarray, params: HBVParameters | np.ndarray) -> np.ndarray:
    """
    Observation function for the HBV model.

    Returns the simulated runoff as a 1 x EnsembleSize array.
    """
    states_array = np.asarray(states, dtype=float)
    if states_array.ndim == 1:
        states_array = states_array.reshape(-1, 1)

    params_array = _as_parameter_vector(params).reshape(-1)
    k0 = params_array[4]
    l = params_array[5]
    k1 = params_array[6]
    k2 = params_array[7]

    s1 = states_array[2:3, :]
    s2 = states_array[3:4, :]
    s1_prev = states_array[4:5, :]

    q = (np.maximum(0.0, s1_prev - l) * k0) + (s1 * k1) + (s2 * k2)
    return q