from .hbvedu import HBVObservation, HBVParameters, HBVTransition
import numpy as np

SJV_Sierra_UBParameters = HBVParameters(
    d=6.0,      # typical HBV range: 0.5 – 8 mm/°C/day, snow basins (Alpine/Sierra analog): 2 – 6
    fc=280.0,   # Alpine shallow soils: 80–250 mm, typical Sierra mixed: 120–300 mm
    beta=5.0,   # humid basins: ~1–3, snowmelt transitional basins: ~2–5
    c=0.04,     # literature HBV-light: ~0.01–0.06 typical, cooler alpine = lower sensitivity
    k0=0.35,    # From HBV manuals + Alpine calibration studies:, fast flow (K0): 0.1–0.4 (days^-1 equivalent conceptual scaling), interflow (K1): 0.02–0.15, baseflow (K2): 0.001–0.05, percolation (Kp): 0.01–0.08, Sierra (steep = faster response):
    l=6.0,      # Alpine HBV: small (2–10 mm equivalent storage index), scaled version: low activation threshold
    k1=0.12,
    k2=0.03,
    kp=0.06,
    pwp=140.0,  # Sierra soils relatively thin/moderate, typical hydrologic modeling range: 80–150 mm
    tt=1.5 # Sierra Nevada: commonly ~ -1 to +2 °C in HBV literature
)
# See comments above, corresponding lower boundary values
SJV_Sierra_LBParameters = HBVParameters(d=2.5, fc=120.0, beta=2, c=0.01, k0=0.15, l=2, k1=0.05, k2=0.005, kp=0.02, pwp=80.0, tt=-0.5)

SJV_Valley_UBParameters = HBVParameters(
    d=5.0,        # Lower importance (less snow contribution locally):
    fc=600.0,     # Deep alluvial soils: Mediterranean HBV: 200–600 mm often calibrated, valley floors higher than alpine by factor ~1.5–2
    beta=7.0,     # Semi-arid basins show strong thresholding: literature: 3–8 typical
    c=0.07,       # Higher temperature sensitivity variability:
    k0=0.2,       # Valley = slower hillslope response, more infiltration losses:
    l=10.0,       # More storage before activation:
    k1=0.08,      # moderate interflow
    k2=0.02,      # meaningful baseflow (fractured rock)
    kp=0.05,      # moderate percolation to deep storage
    pwp=400.0,    # High due to deep soils + vegetation + irrigation buffering:
    tt=2.0        # Same climate physics; valley snow is rare locally
)

# See comments above, corresponding lower boundary values
SJV_Valley_LBParameters = HBVParameters(d=1.5, fc=250.0, beta=3.5, c=0.02, k0=0.05, l=3.0, k1=0.02, k2=0.001, kp=0.01, pwp=150.0, tt=-1.0)

def RainfallCorrTransition(states: np.ndarray, ens_inputs: np.ndarray, params: HBVParameters | np.ndarray,
                           correction: str = 'multiplicative', method: str = 'random_walk', storm_threshold: float | None = None, **kwargs) -> np.ndarray:
    states_array = np.asarray(states, dtype=float)
    if states_array.ndim == 1:
        states_array = states_array.reshape(-1, 1)

    hbv_states = states_array[0:5, :]
    b = states_array[5:6, :]

    if method == 'random_walk':
        get_next_b = lambda b: b
    elif method == 'autoregressive':
        phi = kwargs.get('phi', 0.8)
        get_next_b = lambda b: phi * b

    if correction == 'multiplicative':
        b_next = np.maximum(get_next_b(b), 0)
        if storm_threshold is not None:
            storm_mask = (ens_inputs[1, :] > storm_threshold)
            ens_inputs[1:2, storm_mask] *= 1 + b_next[:, storm_mask]
        else:
            ens_inputs[1:2, :] *= 1 + b

    if correction == 'logmul':
        b_next = get_next_b(b)
        if storm_threshold is not None:
            storm_mask = (ens_inputs[1, :] > storm_threshold)
            ens_inputs[1:2, storm_mask] *= np.exp(b_next[:, storm_mask])
        else:
            ens_inputs[1:2, :] *= np.exp(b)

    if correction == 'additive':
        b_next = get_next_b(b)
        if storm_threshold is not None:
            storm_mask = (ens_inputs[1, :] > storm_threshold)
            ens_inputs[1:2, storm_mask] += b_next[:, storm_mask]
        else:
            ens_inputs[1:2, :] += b_next

    hbv_states_next = HBVTransition(hbv_states, ens_inputs, params)

    return np.vstack([hbv_states_next, b_next])

def RainfallCorrObservation(states: np.ndarray, params: HBVParameters | np.ndarray) -> np.ndarray:
    states_array = np.asarray(states, dtype=float)
    if states_array.ndim == 1:
        states_array = states_array.reshape(-1, 1)

    hbv_states = states_array[0:5, :]
    b = states_array[5:6, :]

    q = HBVObservation(hbv_states, params)

    return np.vstack([q, b])