"""Real-time ocean data fetchers.

Every function in this module pulls a **single day** of real data and
regrids it onto the 0.25 deg Model 2 prediction mesh (73 x 33 cells).

* ERDDAP — no authentication required:
    - SST:           NOAA OISST v2.1 (0.25 deg daily, final product)
    - Chlorophyll:   VIIRS S-NPP+NOAA-20 DINEOF gap-filled daily 4 km
* CMEMS Copernicus Marine — authenticated (env: CMEMS_USERNAME/PASSWORD):
    - SSH:           DUACS NRT L4 0.125 deg daily
    - Salinity:      GLORYS operational analysis, surface depth
    - Dissolved O2:  CMEMS BGC operational analysis, surface depth

The ``fetch_day`` entry point returns a DataFrame with columns
``lat, lon, sst, sst_anomaly, chlorophyll, salinity, dissolved_oxygen, ssh``
containing exactly ``N_LAT * N_LON`` rows (2 409).

Failures are raised, never silently papered over.  Callers may catch and
propagate them to an API error so the user is informed.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import xarray as xr

# copernicusmarine + netCDF4 are not thread-safe across processes, so we
# serialise CMEMS calls through a single lock.  ERDDAP (plain HTTP) can run
# in parallel freely.
_CMEMS_LOCK = threading.Lock()

from ..config import (
    CACHE_DIR,
    CHL_DATASET,
    CHL_VARIABLE,
    CMEMS_OXYGEN_DATASET,
    CMEMS_OXYGEN_VARIABLE,
    CMEMS_PASSWORD,
    CMEMS_SALINITY_DATASET,
    CMEMS_SALINITY_VARIABLE,
    CMEMS_SSH_DATASET,
    CMEMS_SSH_VARIABLE,
    CMEMS_USERNAME,
    ERDDAP_COASTWATCH_BASE,
    ERDDAP_COASTWATCH_NOAA_BASE,
    LAT_MAX,
    LAT_MIN,
    LAT_VALUES,
    LON_MAX,
    LON_MIN,
    LON_VALUES,
    O2_MMOL_M3_TO_ML_L,
    OISST_DATASET,
)

log = logging.getLogger(__name__)


class DataFetchError(RuntimeError):
    """Raised when a required upstream feed cannot be fetched."""


# ---------------------------------------------------------------------------
# Dataset coverage probing (cached per-process)
# ---------------------------------------------------------------------------
_COVERAGE_CACHE: dict[str, date] = {}


def _probe_erddap_end(base: str, dataset: str) -> Optional[date]:
    info_url = base.replace("/griddap", "/info") + f"/{dataset}/index.json"
    try:
        r = requests.get(info_url, timeout=30)
        if r.status_code != 200:
            return None
        rows = r.json().get("table", {}).get("rows", [])
        for row in rows:
            if "time_coverage_end" in row:
                for item in row:
                    if isinstance(item, str) and "T" in item and item[:4].isdigit():
                        return datetime.strptime(item[:10], "%Y-%m-%d").date()
    except Exception as exc:                                           # noqa: BLE001
        log.warning("Could not probe %s/%s: %s", base, dataset, exc)
    return None


def coverage_end(source: str) -> Optional[date]:
    if source in _COVERAGE_CACHE:
        return _COVERAGE_CACHE[source]
    end: Optional[date] = None
    if source == "sst":
        end = _probe_erddap_end(ERDDAP_COASTWATCH_BASE, OISST_DATASET)
    elif source == "chl":
        end = _probe_erddap_end(ERDDAP_COASTWATCH_NOAA_BASE, CHL_DATASET)
    if end is not None:
        _COVERAGE_CACHE[source] = end
    return end


def latest_available_day() -> date:
    """Return the latest date for which *every* mandatory feed has data.

    We probe the ERDDAP datasets (cheap) and assume CMEMS NRT lags by at
    most 2 days beyond SST.  If a probe fails we fall back to
    ``utcnow() - LATENCY_DAYS`` as a safe default.
    """
    from ..config import LATENCY_DAYS

    sst_end = coverage_end("sst")
    chl_end = coverage_end("chl")
    candidates = [d for d in (sst_end, chl_end) if d is not None]
    if not candidates:
        return datetime.utcnow().date() - timedelta(days=LATENCY_DAYS)
    ocean_end = min(candidates)
    # Leave a one-day safety margin - some NRT products update overnight.
    return ocean_end - timedelta(days=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cache_path(name: str, day: date) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_{day.isoformat()}.nc"


def _http_download(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
                if r.status_code >= 400:
                    raise DataFetchError(f"HTTP {r.status_code} for {url}")
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            if dest.stat().st_size == 0:
                raise DataFetchError(f"Empty response body for {url}")
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
    assert last_exc is not None
    raise DataFetchError(f"Failed to download {url}: {last_exc}") from last_exc


def _regrid_to_master(da: xr.DataArray) -> np.ndarray:
    """Interpolate ``da`` (lat, lon) onto the master 0.25 deg grid."""
    # Ensure canonical dim names
    rename = {}
    for src in ("latitude", "y"):
        if src in da.dims:
            rename[src] = "lat"
    for src in ("longitude", "x"):
        if src in da.dims:
            rename[src] = "lon"
    if rename:
        da = da.rename(rename)

    # Drop singleton depth/altitude/time dims
    for extra in ("depth", "altitude", "zlev", "elevation"):
        if extra in da.dims:
            da = da.isel({extra: 0}, drop=True)
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.isel(time=0, drop=True)

    # xarray requires monotonically increasing coords for interp
    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")
    if da["lon"].values[0] > da["lon"].values[-1]:
        da = da.sortby("lon")

    interp = da.interp(lat=LAT_VALUES, lon=LON_VALUES, method="linear")
    return np.asarray(interp.values, dtype="float32")


# ---------------------------------------------------------------------------
# ERDDAP - SST
# ---------------------------------------------------------------------------
def _erddap_query(
    base: str,
    dataset: str,
    variable: str,
    day: date,
    *,
    extra_dims: list[str] | None = None,
    lat_desc: bool = False,
) -> str:
    t = f"{day.isoformat()}T12:00:00Z"
    lat_slice = (
        f"({LAT_MAX + 0.25}):1:({LAT_MIN - 0.25})"
        if lat_desc
        else f"({LAT_MIN - 0.25}):1:({LAT_MAX + 0.25})"
    )
    lon_slice = f"({LON_MIN - 0.25}):1:({LON_MAX + 0.25})"
    query = f"{variable}[({t}):1:({t})]"
    for dim in extra_dims or []:
        query += dim
    query += f"[{lat_slice}][{lon_slice}]"
    return f"{base}/{dataset}.nc?{quote(query, safe=',:/')}"


OISST_FINAL_DATASET = "ncdcOisst21Agg_LonPM180"


def _oisst_url(day: date, dataset: str | None = None) -> str:
    # OISST has a zlev dim: [(0.0):1:(0.0)]
    return _erddap_query(
        ERDDAP_COASTWATCH_BASE, dataset or OISST_DATASET, "sst", day,
        extra_dims=["[(0.0):1:(0.0)]"],
    )


def fetch_sst_day(day: date, dataset: str | None = None) -> np.ndarray:
    dataset = dataset or OISST_DATASET
    tag = "oisst_nrt" if dataset == OISST_DATASET else "oisst_final"
    cache = _cache_path(tag, day)
    if not cache.exists() or cache.stat().st_size == 0:
        log.info("Downloading %s SST for %s", tag, day)
        _http_download(_oisst_url(day, dataset), cache)
    ds = xr.open_dataset(cache)
    try:
        sst = ds["sst"]
        return _regrid_to_master(sst)
    finally:
        ds.close()


def fetch_sst_climatology(day_of_year: int) -> np.ndarray:
    """Return the mean OISST for ``day_of_year`` across 2018..2024.

    Used to compute the SST anomaly feature that Model 2 expects.  The
    climatology is lazily built per DOY and persisted in ``CACHE_DIR``.
    """
    cache = CACHE_DIR / f"oisst_clim_doy{day_of_year:03d}.npy"
    if cache.exists():
        return np.load(cache)

    # 7-year window is enough to stabilise the DOY mean and keeps bootstrap fast.
    reference_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    from concurrent.futures import ThreadPoolExecutor

    def _one(yr: int) -> np.ndarray | None:
        try:
            clim_day = datetime.strptime(f"{yr}-{day_of_year:03d}", "%Y-%j").date()
        except ValueError:
            return None
        try:
            return fetch_sst_day(clim_day, dataset=OISST_FINAL_DATASET)
        except DataFetchError as exc:
            log.warning("Climatology skip %s (%s)", clim_day, exc)
            return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        stack = [r for r in ex.map(_one, reference_years) if r is not None]

    if not stack:
        raise DataFetchError(
            f"No SST climatology could be built for DOY {day_of_year}"
        )
    clim = np.nanmean(np.stack(stack), axis=0).astype("float32")
    np.save(cache, clim)
    return clim


# ---------------------------------------------------------------------------
# ERDDAP - Chlorophyll (VIIRS DINEOF gap-filled)
# ---------------------------------------------------------------------------
def _chl_url(day: date) -> str:
    # VIIRS DINEOF chlorophyll has an altitude dim at the start
    return _erddap_query(
        ERDDAP_COASTWATCH_NOAA_BASE, CHL_DATASET, CHL_VARIABLE, day,
        extra_dims=["[(0.0):1:(0.0)]"],
        lat_desc=True,
    )


def fetch_chlorophyll_day(day: date) -> np.ndarray:
    cache = _cache_path("viirs_chl", day)
    if not cache.exists() or cache.stat().st_size == 0:
        log.info("Downloading VIIRS chlorophyll for %s", day)
        _http_download(_chl_url(day), cache)
    ds = xr.open_dataset(cache)
    try:
        chl = ds[CHL_VARIABLE]
        return _regrid_to_master(chl)
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# CMEMS helpers
# ---------------------------------------------------------------------------
def _cmems_subset(
    dataset_id: str,
    variable: str,
    day: date,
    cache: Path,
    needs_depth: bool = False,
) -> None:
    import copernicusmarine as cm

    if not CMEMS_USERNAME or not CMEMS_PASSWORD:
        raise DataFetchError(
            "CMEMS credentials missing - set CMEMS_USERNAME and CMEMS_PASSWORD."
        )

    kwargs = dict(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_longitude=LON_MIN - 0.5,
        maximum_longitude=LON_MAX + 0.5,
        minimum_latitude=LAT_MIN - 0.5,
        maximum_latitude=LAT_MAX + 0.5,
        start_datetime=f"{day.isoformat()}T00:00:00",
        end_datetime=f"{day.isoformat()}T23:59:59",
        output_filename=cache.name,
        output_directory=str(cache.parent),
        username=CMEMS_USERNAME,
        password=CMEMS_PASSWORD,
        overwrite=True,
    )
    if needs_depth:
        kwargs["minimum_depth"] = 0.0
        kwargs["maximum_depth"] = 1.0
    try:
        with _CMEMS_LOCK:
            cm.subset(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError(f"CMEMS subset failed for {dataset_id} on {day}: {exc}") from exc


def fetch_ssh_day(day: date) -> np.ndarray:
    cache = _cache_path("cmems_ssh", day)
    if not cache.exists() or cache.stat().st_size == 0:
        _cmems_subset(CMEMS_SSH_DATASET, CMEMS_SSH_VARIABLE, day, cache)
    ds = xr.open_dataset(cache)
    try:
        return _regrid_to_master(ds[CMEMS_SSH_VARIABLE])
    finally:
        ds.close()


def fetch_salinity_day(day: date) -> np.ndarray:
    cache = _cache_path("cmems_sal", day)
    if not cache.exists() or cache.stat().st_size == 0:
        _cmems_subset(CMEMS_SALINITY_DATASET, CMEMS_SALINITY_VARIABLE, day, cache, needs_depth=True)
    ds = xr.open_dataset(cache)
    try:
        return _regrid_to_master(ds[CMEMS_SALINITY_VARIABLE])
    finally:
        ds.close()


def fetch_oxygen_day(day: date) -> np.ndarray:
    cache = _cache_path("cmems_o2", day)
    if not cache.exists() or cache.stat().st_size == 0:
        _cmems_subset(CMEMS_OXYGEN_DATASET, CMEMS_OXYGEN_VARIABLE, day, cache, needs_depth=True)
    ds = xr.open_dataset(cache)
    try:
        o2 = _regrid_to_master(ds[CMEMS_OXYGEN_VARIABLE])
        return o2 * O2_MMOL_M3_TO_ML_L     # convert to ml/L to match training units
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Single-day assembler
# ---------------------------------------------------------------------------
def fetch_day(day: date) -> pd.DataFrame:
    """Return a DataFrame with every variable for one day on the master grid.

    The returned frame has ``N_LAT * N_LON`` = 2 409 rows and the columns

        time, lat, lon, sst, sst_anomaly, chlorophyll, salinity,
        dissolved_oxygen, ssh
    """
    log.info("Assembling full ocean state for %s", day)
    sst = fetch_sst_day(day)
    clim = fetch_sst_climatology(day.timetuple().tm_yday)
    sst_anomaly = sst - clim
    chl = fetch_chlorophyll_day(day)
    sal = fetch_salinity_day(day)
    do = fetch_oxygen_day(day)
    ssh = fetch_ssh_day(day)

    lats, lons = np.meshgrid(LAT_VALUES, LON_VALUES, indexing="ij")
    frame = pd.DataFrame(
        {
            "time": pd.Timestamp(day),
            "lat": lats.ravel(),
            "lon": lons.ravel(),
            "sst": sst.ravel(),
            "sst_anomaly": sst_anomaly.ravel(),
            "chlorophyll": chl.ravel(),
            "salinity": sal.ravel(),
            "dissolved_oxygen": do.ravel(),
            "ssh": ssh.ravel(),
        }
    )
    return frame
