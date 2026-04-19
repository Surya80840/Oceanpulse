"""Builds the 13-column feature matrix consumed by Model 2.

The column order must match what was produced during training (see
``new_species_model/build_species_distribution_model.py::build_features``):

    lat, lon, sst, sst_anomaly, chlorophyll_log1p, salinity, dissolved_oxygen,
    ssh, day_of_year_sin, day_of_year_cos, month_sin, month_cos, year
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "lat",
    "lon",
    "sst",
    "sst_anomaly",
    "chlorophyll_log1p",
    "salinity",
    "dissolved_oxygen",
    "ssh",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
    "year",
]


def build_features(forecast: pd.DataFrame) -> pd.DataFrame:
    """Turn a Model-1 forecast frame into Model-2 ready features."""
    work = forecast.copy()
    doy_angle = 2.0 * np.pi * work["day_of_year"].astype(float) / 365.25
    month_angle = 2.0 * np.pi * work["month"].astype(float) / 12.0

    features = pd.DataFrame(
        {
            "lat": work["lat"].astype(float),
            "lon": work["lon"].astype(float),
            "sst": work["sst"].astype(float),
            "sst_anomaly": work["sst_anomaly"].astype(float),
            "chlorophyll_log1p": np.log1p(work["chlorophyll"].astype(float)),
            "salinity": work["salinity"].astype(float),
            "dissolved_oxygen": work["dissolved_oxygen"].astype(float),
            "ssh": work["ssh"].astype(float),
            "day_of_year_sin": np.sin(doy_angle),
            "day_of_year_cos": np.cos(doy_angle),
            "month_sin": np.sin(month_angle),
            "month_cos": np.cos(month_angle),
            "year": work["year"].astype(float),
        }
    )
    features.index = work.index
    return features[FEATURE_COLUMNS]
