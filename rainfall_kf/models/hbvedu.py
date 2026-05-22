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

def HBVTransition(states: np.ndarray, ens_inputs: np.ndarray, params: HBVParameters | np.ndarray) -> np.ndarray:
    """
    Transition function for the HBV rainfall-runoff model.

    The state and input arrays follow the same column-vector convention as the
    MATLAB code, but vectorized across ensemble members.
    """
    states_array = np.asarray(states, dtype=float)
    if states_array.ndim == 1:
        states_array = states_array.reshape(-1, 1)

    params_array = _as_parameter_vector(params).reshape(-1)

    snow = states_array[0:1, :]
    soil = states_array[1:2, :]
    s1 = states_array[2:3, :]
    s2 = states_array[3:4, :]

    temp = ens_inputs[0:1, :]
    prec = ens_inputs[1:2, :]
    temp_m = ens_inputs[2:3, :]
    dpem = ens_inputs[3:4, :]

    d, fc, beta, c, k0, l, k1, k2, kp, pwp, tt = params_array

    below_threshold = temp < tt
    melt = d * (temp - tt)
    soil_nonnegative = np.maximum(soil, 0.0)

    snow_next = np.where(below_threshold, snow + prec, np.maximum(snow - melt, 0.0))
    lwater = np.where(below_threshold, 0.0, prec + np.minimum(snow, melt))

    pe = np.maximum((1.0 + c * (temp - temp_m)) * dpem, 0.0)
    ea = np.where(soil > pwp, pe, pe * (soil_nonnegative / pwp))
    dq = lwater * ((soil_nonnegative / fc) ** beta)
    dq = np.clip(dq, 0.0, lwater)

    s1_next = np.maximum(s1 + dq - (np.maximum(0.0, s1 - l) * k0) - (s1 * k1) - (s1 * kp), 0.0)
    s2_next = np.maximum(s2 + (s1 * kp) - (s2 * k2), 0.0)
    soil_next = np.maximum(soil + lwater - dq - ea, 0.0)

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