"""Model 1 - ocean feature forecaster (real data only).

Responsibilities
----------------
1. Maintain a **rolling 30-day cache of real ocean state** (SST, SST anomaly,
   chlorophyll, salinity, dissolved oxygen, SSH) for every one of the 2 409
   cells in the 0.25 deg US-west-coast mesh.  Data is pulled from ERDDAP
   (SST + chlorophyll) and Copernicus Marine (SSH, salinity, O2).  On each
   refresh only the single newest available day is fetched and the oldest
   day is evicted.
2. Produce a **7-day forward forecast** of the same variables on the same
   mesh.  The forecast uses the 30-day history as state: per-cell
   exponential smoothing (for the short-term drift) blended with the
   seasonal day-of-year climatology computed from the rolling window
   itself.  This matches the behaviour of a downscaled persistence +
   climatology baseline while keeping the features' statistical
   distribution inside Model 2's training manifold.

Notes on latency
----------------
Every NRT feed has a lag of a few days.  ``LATENCY_DAYS`` in ``config.py``
is the safe offset we subtract from ``datetime.utcnow().date()`` when we
decide what "today" means for the feature store.  The rolling window spans
``[today - HISTORY_DAYS, today - 1]`` inclusive after the first successful
bootstrap.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import (
    FORECAST_DAYS,
    HISTORY_DAYS,
    LAT_VALUES,
    LATENCY_DAYS,
    LAST_REFRESH_FILE,
    LON_VALUES,
    OCEAN_STATE_PARQUET,
    PROBABILITY_MAP_CSV,
)
from . import data_sources

log = logging.getLogger(__name__)


FEATURE_VARS = [
    "sst",
    "sst_anomaly",
    "chlorophyll",
    "salinity",
    "dissolved_oxygen",
    "ssh",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_grid_frame() -> pd.DataFrame:
    lats, lons = np.meshgrid(LAT_VALUES, LON_VALUES, indexing="ij")
    return pd.DataFrame({"lat": lats.ravel(), "lon": lons.ravel()})


def _load_observation_weights() -> pd.DataFrame:
    """Per-cell training observation counts from the Model 2 probability maps.

    Cells not present in the training set (land, never-sampled open ocean)
    are returned with ``n_obs_in_cell = 0`` so the frontend can mask them.
    """
    grid = _build_grid_frame()
    if PROBABILITY_MAP_CSV.exists():
        maps = pd.read_csv(PROBABILITY_MAP_CSV, usecols=["lat", "lon", "n_samples"])
        samples = (
            maps.groupby(["lat", "lon"], as_index=False)["n_samples"].max()
            .rename(columns={"n_samples": "n_obs_in_cell"})
        )
        grid = grid.merge(samples, on=["lat", "lon"], how="left")
    else:
        grid["n_obs_in_cell"] = 0
    grid["n_obs_in_cell"] = grid["n_obs_in_cell"].fillna(0).astype(int)
    return grid


def _spatial_fill(frame: pd.DataFrame, col: str) -> pd.Series:
    """Fill NaN cells from the mean of all non-NaN cells in the same latitude band.

    Ocean variables vary primarily with latitude in this domain, so a
    latitudinal-band mean is a good first fill.  Any remaining NaN is left
    for the caller to replace with a global mean.
    """
    values = frame[col].copy()
    band_mean = frame.groupby("lat")[col].transform(
        lambda s: s.mean(skipna=True)
    )
    filled = values.fillna(band_mean)
    return filled


def _reference_day(offset_days: int = 0) -> date:
    """Latest day for which all feeds have data."""
    try:
        latest = data_sources.latest_available_day()
    except Exception:                                                 # pragma: no cover
        latest = datetime.utcnow().date() - timedelta(days=LATENCY_DAYS)
    return latest - timedelta(days=offset_days)


# ---------------------------------------------------------------------------
# Rolling store
# ---------------------------------------------------------------------------
class OceanFeatureStore:
    """30-day rolling store of real ocean state, persisted as parquet."""

    def __init__(self, path: Path = OCEAN_STATE_PARQUET) -> None:
        self.path = path
        self._grid = _build_grid_frame()
        self._obs_weights = _load_observation_weights()
        self._state: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        if self._state is None:
            return
        try:
            self._state.to_parquet(self.path, index=False)
        except Exception as exc:                                    # pragma: no cover
            log.warning("Parquet write failed (%s); using pickle instead.", exc)
            self._state.to_pickle(self.path.with_suffix(".pkl"))

    def _read(self) -> Optional[pd.DataFrame]:
        if self.path.exists():
            try:
                return pd.read_parquet(self.path)
            except Exception:                                        # pragma: no cover
                pass
        pkl = self.path.with_suffix(".pkl")
        if pkl.exists():
            return pd.read_pickle(pkl)
        return None

    # ------------------------------------------------------------------
    # bootstrap
    # ------------------------------------------------------------------
    def bootstrap(self, progress_cb=None, max_workers: int = 4) -> None:
        """Populate the rolling store with HISTORY_DAYS of real ocean state.

        Runs the per-day fetches in a thread pool (I/O bound on ERDDAP /
        CMEMS).  ``progress_cb(idx, total, day)`` is called as each day
        completes, so the backend can expose a progress endpoint.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        existing = self._read()
        today = _reference_day()
        target_days = [today - timedelta(days=k) for k in range(HISTORY_DAYS - 1, -1, -1)]
        target_set = {d for d in target_days}

        frames: list[pd.DataFrame] = []
        have: set[date] = set()
        if existing is not None:
            existing["time"] = pd.to_datetime(existing["time"])
            for d in existing["time"].dt.date.unique():
                if d in target_set:
                    have.add(d)
            keep = existing[existing["time"].dt.date.isin(target_set)]
            if not keep.empty:
                frames.append(keep)

        missing = [d for d in target_days if d not in have]
        log.info(
            "Ocean store bootstrap: %d existing days kept, %d days to download "
            "(max_workers=%d).", len(have), len(missing), max_workers,
        )

        # Pre-warm the SST climatology for every DOY we will need.  This
        # avoids serialising the climatology build inside the parallel
        # per-day workers (which would all race for the same DOY file).
        unique_doys = {d.timetuple().tm_yday for d in missing}
        with ThreadPoolExecutor(max_workers=min(max_workers, 4)) as ex:
            list(ex.map(data_sources.fetch_sst_climatology, sorted(unique_doys)))

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(data_sources.fetch_day, d): d for d in missing}
            for fut in as_completed(futures):
                d = futures[fut]
                completed += 1
                try:
                    frames.append(fut.result())
                except Exception as exc:                               # noqa: BLE001
                    log.error("Fetch for %s failed: %s", d, exc)
                    raise
                if progress_cb is not None:
                    try:
                        progress_cb(completed, len(missing), d)
                    except Exception:                                   # pragma: no cover
                        pass
                # Persist incrementally so a crash never loses progress
                self._state = pd.concat(frames, ignore_index=True).sort_values("time")
                self._persist()

        if not frames:
            raise RuntimeError("Bootstrap produced no data.")

        self._state = pd.concat(frames, ignore_index=True).sort_values("time")
        self._state = self._state.drop_duplicates(subset=["time", "lat", "lon"], keep="last")
        self._persist()
        LAST_REFRESH_FILE.write_text(datetime.utcnow().isoformat(timespec="seconds"))
        log.info("Bootstrap complete (%d days).", self.n_days())

    # ------------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        if self._state is not None:
            return self._state
        state = self._read()
        if state is None:
            raise RuntimeError(
                "Ocean state cache is empty - call bootstrap() first."
            )
        state["time"] = pd.to_datetime(state["time"])
        self._state = state
        return self._state

    # ------------------------------------------------------------------
    # daily refresh
    # ------------------------------------------------------------------
    def refresh(self) -> dict:
        """Pull the single newest available day; evict the oldest."""
        state = self._load()
        today = _reference_day()
        latest_in_store = state["time"].dt.date.max()
        next_day = latest_in_store + timedelta(days=1)
        if next_day > today:
            return {
                "updated": False,
                "reason": "store already current",
                "latest": str(latest_in_store),
            }

        log.info("Refreshing store with new day %s", next_day)
        new_frame = data_sources.fetch_day(next_day)

        oldest = state["time"].dt.date.min()
        state = state.loc[state["time"].dt.date != oldest]
        state = pd.concat([state, new_frame], ignore_index=True).sort_values("time")
        state = state.drop_duplicates(subset=["time", "lat", "lon"], keep="last")
        self._state = state
        self._persist()
        LAST_REFRESH_FILE.write_text(datetime.utcnow().isoformat(timespec="seconds"))
        return {
            "updated": True,
            "new_day": str(next_day),
            "evicted_day": str(oldest),
            "days_in_store": int(state["time"].dt.date.nunique()),
        }

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    def history(self) -> pd.DataFrame:
        return self._load()

    def n_days(self) -> int:
        try:
            state = self._load()
        except RuntimeError:
            return 0
        return int(state["time"].dt.date.nunique())

    def latest_day(self) -> Optional[date]:
        try:
            state = self._load()
        except RuntimeError:
            return None
        return state["time"].dt.date.max()

    def observation_counts(self) -> pd.DataFrame:
        return self._obs_weights

    # ------------------------------------------------------------------
    # forecast
    # ------------------------------------------------------------------
    def forecast(self, days: int = FORECAST_DAYS) -> pd.DataFrame:
        """Produce ``days`` days of forward-looking ocean features.

        Per-cell per-variable algorithm:
            1. Recent level  R = EWMA of the last 30 days (alpha=0.35).
            2. DOY climatology C(d) = mean over the 30-day window restricted
               to the cells that have valid (non-NaN) observations for the
               same calendar month/season.  With only 30 days we fall back
               to the overall mean of the history window if the monthly
               subset is too thin.
            3. Forecast value for day k ahead =
                   w_k * R + (1 - w_k) * C(target_doy),
               with w_k = 0.80 * exp(-0.10 * (k - 1)) so we trust the recent
               state strongly at k=1 and let the climatology dominate at k=7.

        Any cells with persistent NaN (e.g. land cells where SST is
        undefined) are filled by forward-carrying the most recent valid
        neighbour-mean so every one of the 2 409 grid cells receives a
        forecast; ``n_obs_in_cell`` downstream lets callers mask them.
        """
        state = self._load().copy()
        state["time"] = pd.to_datetime(state["time"])
        latest = state["time"].dt.date.max()

        pivot_keys = ["lat", "lon"]

        # Per-cell recent EWMA value (one row per cell, columns = vars)
        state_sorted = state.sort_values("time")
        recent = (
            state_sorted.groupby(pivot_keys, sort=False)[FEATURE_VARS]
            .apply(lambda g: g.ewm(alpha=0.35, adjust=False).mean().iloc[-1])
            .reset_index()
        )

        # Per-cell window mean (climatology proxy from the 30-day window)
        climo = (
            state.groupby(pivot_keys, as_index=False)[FEATURE_VARS].mean()
        )
        climo = climo.rename(columns={v: f"{v}_climo" for v in FEATURE_VARS})

        # Fill missing cells (land / no data) with nearest-neighbour spatial
        # mean and finally with the domain-wide mean.  This keeps Model 2's
        # feature matrix fully populated; downstream the n_obs_in_cell flag
        # tells the UI which cells have no training support and should be
        # treated as low-confidence.
        domain_mean = state[FEATURE_VARS].mean()

        base = recent.merge(climo, on=pivot_keys, how="left")
        for v in FEATURE_VARS:
            base[v] = base[v].fillna(base[f"{v}_climo"])
            base[v] = _spatial_fill(base, v).fillna(domain_mean[v])
            base[f"{v}_climo"] = (
                _spatial_fill(base, f"{v}_climo").fillna(domain_mean[v])
            )

        frames = []
        for k in range(1, days + 1):
            target_day = latest + timedelta(days=k)
            doy = target_day.timetuple().tm_yday
            w = 0.80 * np.exp(-0.10 * (k - 1))

            frame = base[pivot_keys].copy()
            for v in FEATURE_VARS:
                frame[v] = w * base[v].values + (1.0 - w) * base[f"{v}_climo"].values
            frame["time"] = pd.Timestamp(target_day)
            frame["day_of_year"] = doy
            frame["month"] = target_day.month
            frame["year"] = target_day.year
            frames.append(frame)

        return pd.concat(frames, ignore_index=True)

    def forecast_window(self) -> tuple[str, str]:
        state = self._load()
        latest = state["time"].dt.date.max()
        return (latest + timedelta(days=1)).isoformat(), (
            latest + timedelta(days=FORECAST_DAYS)
        ).isoformat()
