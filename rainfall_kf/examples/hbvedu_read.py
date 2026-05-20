from pathlib import Path

import numpy as np


_HBV_DATA_DIR = Path(__file__).resolve().parents[0] / "HBV_EDU"


def _resolve_path(filepath: str | Path | None, default_name: str) -> Path:
    if filepath is None:
        return _HBV_DATA_DIR / default_name
    return Path(filepath)


def read_hbv_inputs(filepath_precip_temp: str | Path | None = None, filepath_monthly_temp_evap: str | Path | None = None) -> np.ndarray:
    """
    Read the HBV input forcing data.

    Returns a 4 x N array in the order [Temp; prec; Temp_m; dpem].
    """
    precip_temp_path = _resolve_path(filepath_precip_temp, "inputPrecipTemp.txt")
    monthly_path = _resolve_path(filepath_monthly_temp_evap, "inputMonthlyTempEvap.txt")

    month_list = []
    temps = []
    precs = []

    with precip_temp_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            month_list.append(int(parts[1]))
            temps.append(float(parts[2]))
            precs.append(float(parts[3]))

    monthly = np.loadtxt(monthly_path, dtype=float, usecols=(0, 1, 2))
    if monthly.ndim == 1:
        monthly = monthly.reshape(1, -1)

    month_list_array = np.asarray(month_list, dtype=int)
    temps_array = np.asarray(temps, dtype=float)
    precs_array = np.asarray(precs, dtype=float)

    temp_m = monthly[month_list_array, 0]
    dpem = monthly[month_list_array, 2]

    return np.vstack([
        temps_array,
        precs_array,
        temp_m,
        dpem,
    ])


def read_hbv_params(filepath_bv: str | Path | None = None, filepath_iv: str | Path | None = None, alpha: float = 0.5) -> np.ndarray:
    """
    Read the HBV parameter bounds and initial values.

    Returns an 11 x 1 array in the order [d; fc; beta; c; k0; l; k1; k2; kp; pwp; tt].
    """
    bv_path = _resolve_path(filepath_bv, "BV.txt")
    iv_path = _resolve_path(filepath_iv, "IV.txt")

    bounds = np.loadtxt(bv_path, dtype=float, usecols=(1, 2))
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    params = alpha * upper + (1.0 - alpha) * lower

    iv_values = np.loadtxt(iv_path, dtype=float, usecols=(1,))
    if iv_values.ndim == 0:
        iv_values = np.asarray([iv_values], dtype=float)
    tt = float(iv_values[1])

    return np.concatenate([params, np.asarray([tt], dtype=float)]).reshape(-1, 1)


def read_hbv_qobs(filepath_qobs: str | Path | None = None) -> np.ndarray:
    """Read the observed runoff time series as a 1 x N array."""
    qobs_path = _resolve_path(filepath_qobs, "Qobs.txt")
    qobs = np.loadtxt(qobs_path, dtype=float)
    return np.asarray(qobs, dtype=float).reshape(1, -1)


def read_hbv_initial(filepath_iv: str | Path | None = None) -> np.ndarray:
    """
    Read the initial HBV state.

    Returns a 5 x 1 array in the order [snow; soil; s1; s2; s1_prev].
    """
    iv_path = _resolve_path(filepath_iv, "IV.txt")

    values = np.loadtxt(iv_path, dtype=float, usecols=(1,))
    if values.ndim == 0:
        values = np.asarray([values], dtype=float)

    snow = values[3]
    soil = values[4]
    s1 = values[5]
    s2 = values[6]
    s1_prev = 0.0

    return np.asarray([[snow], [soil], [s1], [s2], [s1_prev]], dtype=float)