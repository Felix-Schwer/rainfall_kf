from dataclasses import dataclass, fields as dataclass_fields
import operator

import numpy as np
import xarray as xr
import pandas as pd

class ParameterSet:
    def _apply(self, other, op):
        if isinstance(other, ParameterSet):
            values = {
                field.name: op(getattr(self, field.name), getattr(other, field.name))
                for field in dataclass_fields(self)
            }
        else:
            values = {
                field.name: op(getattr(self, field.name), other)
                for field in dataclass_fields(self)
            }
        return type(self)(**values)

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, field.name) for field in dataclass_fields(self)], dtype=float)

    def __add__(self, other):
        return self._apply(other, operator.add)

    def __radd__(self, other):
        return self._apply(other, operator.add)

    def __sub__(self, other):
        return self._apply(other, operator.sub)

    def __rsub__(self, other):
        if isinstance(other, ParameterSet):
            return other._apply(self, operator.sub)
        return type(self)(**{
            field.name: operator.sub(other, getattr(self, field.name))
            for field in dataclass_fields(self)
        })

    def __mul__(self, other):
        return self._apply(other, operator.mul)

    def __rmul__(self, other):
        return self._apply(other, operator.mul)

    def __truediv__(self, other):
        return self._apply(other, operator.truediv)

    def __rtruediv__(self, other):
        if isinstance(other, ParameterSet):
            return other._apply(self, operator.truediv)
        return type(self)(**{
            field.name: operator.truediv(other, getattr(self, field.name))
            for field in dataclass_fields(self)
        })


@dataclass()
class HBVParameters(ParameterSet):
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

def monthly_longterm_inputs_fromds(et_ds: xr.Dataset, temp_ds: xr.Dataset, starttime: str | pd.Timestamp, endtime: str | pd.Timestamp) -> tuple[xr.DataArray, xr.DataArray]:
    time_range = pd.date_range(start=starttime, end=endtime, freq="D")
    et_ds["et"] = et_ds["et"] / et_ds.time.dt.days_in_month
    et_ds = (et_ds.reindex(time=time_range, method="ffill"))
    et_ds.et.attrs["units"] = "mm/day"
    et_ds.et.attrs["description"] = "Basin-averaged evapotranspiration from OpenET. Simple mean over all data points within each HUC8 basin. Resampled to daily resolution by forward-filling monthly values."
    et_ds.et.attrs["long_name"] = "Daily downsampled basin-averaged evapotranspiration"

    monthly_temp = temp_ds.tmean.groupby("time.month").mean(dim="time")
    monthly_temp = monthly_temp.sel(month=temp_ds.time.dt.month)
    monthly_temp = monthly_temp.assign_coords(time=temp_ds.time)
    monthly_temp = monthly_temp.drop_vars("month")
    monthly_temp.attrs["description"] = "Monthly long term basin-averaged mean temperature from PRISM"
    monthly_temp.attrs["long_name"] = "Monthly long term basin-averaged mean temperature"

    monthly_et = et_ds.et.groupby("time.month").mean(dim="time")
    monthly_et = monthly_et.sel(month=et_ds.time.dt.month)
    monthly_et = monthly_et.assign_coords(time=et_ds.time)
    monthly_et = monthly_et.drop_vars("month")
    monthly_et.attrs["description"] = "Monthly long term basin-averaged evapotranspiration from OpenET"
    monthly_et.attrs["long_name"] = "Monthly long term basin-averaged evapotranspiration"

    return monthly_et, monthly_temp

def _as_parameter_vector(params: ParameterSet | np.ndarray) -> np.ndarray:
    if isinstance(params, ParameterSet):
        return params.as_array().reshape(-1, 1)

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

    snow = np.maximum(states_array[0:1, :], 0.0)
    soil = np.maximum(states_array[1:2, :], 0.0)
    s1 =  np.maximum(states_array[2:3, :], 0.0)
    s2 = np.maximum(states_array[3:4, :], 0.0)

    temp = ens_inputs[0:1, :]
    prec = ens_inputs[1:2, :]
    temp_m = ens_inputs[2:3, :]
    dpem = ens_inputs[3:4, :]

    d, fc, beta, c, k0, l, k1, k2, kp, pwp, tt = params_array

    below_threshold = temp < tt
    melt = d * np.maximum(temp - tt, 0.0)

    snow_next = np.where(below_threshold, snow + prec, np.maximum(snow - melt, 0.0))
    lwater = prec + np.minimum(snow, melt)

    pe = np.maximum((1.0 + c * (temp - temp_m)) * dpem, 0.0)
    ea = np.where(soil >= pwp, pe, pe * (soil / pwp))
    dq = lwater * (np.clip(soil / fc, 0.0, 1.0) ** beta)

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