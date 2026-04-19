"""End-to-end Model 1 -> Model 2 pipeline.

* Pulls the next-7-day ocean forecast from the rolling store.
* Builds Model-2 features.
* Scores every species and caches the resulting (cell x species) probability
  surface in memory.  The surface is invalidated whenever the feature store
  receives a new day of data.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import pandas as pd

from ..config import FORECAST_DAYS, TOP_K
from .feature_builder import build_features
from .model_loader import SpeciesDistributionModel
from .ocean_forecaster import OceanFeatureStore

log = logging.getLogger(__name__)


def _pretty_name(species: str) -> str:
    return species.replace("_", " ").strip().title()


class SpeciesForecastPipeline:
    def __init__(
        self,
        model: SpeciesDistributionModel,
        store: OceanFeatureStore,
    ) -> None:
        self.model = model
        self.store = store
        self._lock = threading.Lock()
        self._cache_version: Optional[str] = None
        self._cell_mean: Optional[pd.DataFrame] = None  # (lat, lon, species) -> mean prob over the 7-day forecast
        self._species_global_mean: Optional[pd.Series] = None
        self._obs = store.observation_counts()

    # ------------------------------------------------------------------
    def species_list(self) -> list[dict]:
        return [
            {"id": sp, "display_name": _pretty_name(sp)}
            for sp in sorted(self.model.species_columns)
        ]

    # ------------------------------------------------------------------
    def _forecast_key(self, forecast: pd.DataFrame) -> str:
        t_min = forecast["time"].min()
        t_max = forecast["time"].max()
        return f"{t_min.date()}..{t_max.date()}"

    def _compute_cell_mean(self) -> pd.DataFrame:
        """Run Model1 -> Model2 for every cell and cache mean probability."""
        forecast = self.store.forecast(FORECAST_DAYS)
        features = build_features(forecast)

        # model.predict_all returns (rows x species) - rows are the 7-day x 2409-cell stack.
        prob = self.model.predict_all(features)
        prob["lat"] = forecast["lat"].values
        prob["lon"] = forecast["lon"].values

        species_cols = self.model.species_columns
        mean_by_cell = prob.groupby(["lat", "lon"], as_index=False)[species_cols].mean()
        self._cache_version = self._forecast_key(forecast)
        return mean_by_cell

    def _ensure_cache(self) -> pd.DataFrame:
        with self._lock:
            forecast = self.store.forecast(FORECAST_DAYS)
            key = self._forecast_key(forecast)
            if self._cell_mean is None or self._cache_version != key:
                log.info("Recomputing per-cell probability surface (key=%s)", key)
                self._cell_mean = self._compute_cell_mean()
                species_cols = self.model.species_columns
                self._species_global_mean = self._cell_mean[species_cols].mean()
            return self._cell_mean

    def invalidate(self) -> None:
        with self._lock:
            self._cell_mean = None
            self._species_global_mean = None
            self._cache_version = None

    # ------------------------------------------------------------------
    def query_bbox(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        top_k: int = TOP_K,
    ) -> dict:
        cell_mean = self._ensure_cache()
        mask = (
            (cell_mean["lat"] >= min(lat_min, lat_max))
            & (cell_mean["lat"] <= max(lat_min, lat_max))
            & (cell_mean["lon"] >= min(lon_min, lon_max))
            & (cell_mean["lon"] <= max(lon_min, lon_max))
        )
        inside = cell_mean.loc[mask]
        if inside.empty:
            raise ValueError("No grid cells inside the requested bounding box.")

        species_cols = self.model.species_columns
        # Mean probability across every cell in the bbox, per species.
        species_mean = inside[species_cols].mean()
        species_max = inside[species_cols].max()

        # Rank by a calibrated "lift" score:
        #   lift = bbox mean probability  -  grid-wide mean probability
        # This surfaces species whose bbox probability is elevated relative
        # to the rest of the domain, correcting for the overconfident
        # class_weight='balanced' logistic regressions which otherwise all
        # saturate at 1.0.
        if self._species_global_mean is not None:
            lift = species_mean - self._species_global_mean
        else:
            lift = species_mean.copy()
        ranked = lift.sort_values(ascending=False)
        top = ranked.head(top_k)
        top_species = [
            {
                "species": sp,
                "display_name": _pretty_name(sp),
                "mean_probability": float(species_mean.loc[sp]),
                "max_probability": float(species_max.loc[sp]),
                "n_cells": int(len(inside)),
            }
            for sp in top.index
        ]

        # Heatmap uses the single top species so the map actually shows where
        # the highest-ranked fish is most likely to be.
        heatmap_species = top.index[0] if len(top) else None
        heatmap: list[dict] = []
        if heatmap_species is not None:
            cells = inside[["lat", "lon", heatmap_species]].copy()
            cells = cells.merge(self._obs, on=["lat", "lon"], how="left")
            for row in cells.itertuples(index=False):
                heatmap.append(
                    {
                        "lat": float(row.lat),
                        "lon": float(row.lon),
                        "probability": float(getattr(row, heatmap_species)),
                        "n_obs_in_cell": int(row.n_obs_in_cell or 0),
                    }
                )

        start, end = self.store.forecast_window()
        return {
            "forecast_start": start,
            "forecast_end": end,
            "n_cells": int(len(inside)),
            "top_species": top_species,
            "heatmap": heatmap,
            "heatmap_species": heatmap_species,
        }

    # ------------------------------------------------------------------
    def query_species(self, species: str, top_k: int = TOP_K) -> dict:
        if species not in self.model.species_columns:
            raise KeyError(species)
        cell_mean = self._ensure_cache()
        cells = cell_mean[["lat", "lon", species]].copy()
        cells = cells.merge(self._obs, on=["lat", "lon"], how="left")
        cells["n_obs_in_cell"] = cells["n_obs_in_cell"].fillna(0).astype(int)

        heatmap = [
            {
                "lat": float(r.lat),
                "lon": float(r.lon),
                "probability": float(getattr(r, species)),
                "n_obs_in_cell": int(r.n_obs_in_cell),
            }
            for r in cells.itertuples(index=False)
        ]

        ranked = cells.sort_values(species, ascending=False).head(top_k)
        top_locations = [
            {
                "lat": float(r.lat),
                "lon": float(r.lon),
                "probability": float(getattr(r, species)),
                "n_obs_in_cell": int(r.n_obs_in_cell),
            }
            for r in ranked.itertuples(index=False)
        ]

        start, end = self.store.forecast_window()
        return {
            "species": species,
            "display_name": _pretty_name(species),
            "forecast_start": start,
            "forecast_end": end,
            "top_locations": top_locations,
            "heatmap": heatmap,
        }
