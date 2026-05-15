from setuptools import find_packages, setup


setup(
    name="rainfall_kf",
    version="0.1.0",
    description="Generic Ensemble Kalman Filter tools for rainfall-runoff models",
    packages=find_packages(include=["rainfall_kf", "rainfall_kf.*", "enkf"]),
)