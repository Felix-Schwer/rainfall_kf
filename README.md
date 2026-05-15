# rainfall_kf

Generic Ensemble Kalman Filter scaffolding for hydrologic models such as HBV.

The quickest end-to-end test is the Lorenz-63 example in [notebooks/lorenz_enkf.ipynb](notebooks/lorenz_enkf.ipynb).

## Package Layout

```text
rainfall_kf/
	core/
	models/
	utils/
	examples/
enkf/
```

The `rainfall_kf` package contains the new generic implementation. The legacy
`enkf` namespace is kept as a compatibility alias.

## Next Steps

1. Port your HBV transition and observation functions from MATLAB to Python.
2. Use [notebooks/lorenz_enkf.ipynb](notebooks/lorenz_enkf.ipynb) to validate the filter in a Colab runtime.
3. Add tests for a toy linear model before validating HBV.