"""Global configuration for the OceanPulse web backend.

The prediction grid matches the Model 2 training grid (``peter_code`` branch,
``new_species_model/``): a regular 0.25 degree mesh covering the US west coast.

All external feeds are **real** - ERDDAP for SST + chlorophyll (no auth) and
Copernicus Marine for SSH, salinity and dissolved oxygen (auth required).
There is no synthetic fallback: if a feed is unavailable the caller raises.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent.parent

MODEL_DIR = REPO_ROOT / "new_species_model"
MODEL_BUNDLE_PATH = MODEL_DIR / "species_distribution_model.joblib"
PROBABILITY_MAP_CSV = MODEL_DIR / "species_probability_maps.csv"

DATA_DIR = BACKEND_DIR / "app" / "data"
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OCEAN_STATE_PARQUET = DATA_DIR / "ocean_state_rolling.parquet"
LAST_REFRESH_FILE = DATA_DIR / "last_refresh.txt"

FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

# ---------------------------------------------------------------------------
# Prediction grid (identical to training grid)
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX, LAT_STEP = 30.0, 48.0, 0.25
LON_MIN, LON_MAX, LON_STEP = -124.0, -116.0, 0.25

LAT_VALUES = np.round(np.arange(LAT_MIN, LAT_MAX + 1e-9, LAT_STEP), 2)
LON_VALUES = np.round(np.arange(LON_MIN, LON_MAX + 1e-9, LON_STEP), 2)
N_LAT = len(LAT_VALUES)  # 73
N_LON = len(LON_VALUES)  # 33
N_CELLS = N_LAT * N_LON  # 2409

# ---------------------------------------------------------------------------
# Feature store / forecast horizon
# ---------------------------------------------------------------------------
HISTORY_DAYS = 30       # rolling ocean state window
FORECAST_DAYS = 7       # Model 1 -> Model 2 horizon
TOP_K = 10

# ---------------------------------------------------------------------------
# Latency: NRT products lag the present.  We walk the "today" pointer back
# by LATENCY_DAYS so every fetch lands inside a populated window.
#   - OISST SST        : ~1-3 day lag (finalised); NRT interim product is daily
#   - VIIRS chlorophyll: 1-3 day lag
#   - CMEMS DUACS SSH  : ~5 day lag
#   - GLORYS salinity  : ~1 day lag (analysis), longer for forecast-free
#   - GLOBIO O2        : ~2-5 day lag
# ---------------------------------------------------------------------------
# This is an *initial* guess.  ``data_sources.probe_coverage()`` replaces it
# with the minimum time_coverage_end we actually see across every feed.
LATENCY_DAYS = 5

# ---------------------------------------------------------------------------
# External services - credentials pulled from env.  The app refuses to start
# without CMEMS creds; the three CMEMS feeds (SSH, salinity, DO) are
# mandatory because Model 2 was trained with those four oceanographic
# channels and we do not substitute them with synthetic data.
# ---------------------------------------------------------------------------
CMEMS_USERNAME = os.environ.get("CMEMS_USERNAME")
CMEMS_PASSWORD = os.environ.get("CMEMS_PASSWORD")

# ERDDAP endpoints
ERDDAP_COASTWATCH_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
ERDDAP_COASTWATCH_NOAA_BASE = "https://coastwatch.noaa.gov/erddap/griddap"
# NRT (near-real-time) OISST closes the ~15-day lag of the final AVHRR product
OISST_DATASET = "ncdcOisst21NrtAgg_LonPM180"
# VIIRS S-NPP + NOAA-20 DINEOF gap-filled daily chlorophyll (present-day coverage)
CHL_DATASET = "noaacwNPPN20VIIRSDINEOFDaily"
CHL_VARIABLE = "chlor_a"

# CMEMS dataset IDs (NRT / operational forecast products)
CMEMS_SSH_DATASET = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"
CMEMS_SSH_VARIABLE = "sla"
CMEMS_SALINITY_DATASET = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
CMEMS_SALINITY_VARIABLE = "so"
CMEMS_OXYGEN_DATASET = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
CMEMS_OXYGEN_VARIABLE = "o2"

# Unit conversion: CMEMS O2 is mmol/m^3, training data used ml/L.
# 1 ml/L of O2 = 44.6596 mmol/m^3  (at STP)  =>  ml/L = mmol/m^3 / 44.6596
O2_MMOL_M3_TO_ML_L = 1.0 / 44.6596
