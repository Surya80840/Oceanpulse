# OceanPulse Web App

A full-stack web application that connects the OceanPulse two-stage fish species
forecasting pipeline and exposes it through an interactive map UI.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          OceanPulse Web App                                │
│                                                                            │
│  ERDDAP + Copernicus Marine ─► Model 1 (ocean forecaster)                  │
│  (real live feeds, 30-day      └► 7-day forecast of 6 ocean variables      │
│   rolling cache, 1 day added     at every one of 2 409 0.25° cells         │
│   per refresh)                                                             │
│                                   │                                        │
│                                   ▼                                        │
│                               Model 2 (peter_code/new_species_model)       │
│                               369 logistic-regression classifiers,         │
│                               per-species probability per cell per day     │
│                                   │                                        │
│                ┌──────────────────┴──────────────────┐                     │
│                ▼                                     ▼                     │
│       /api/query/bbox                       /api/query/species             │
│       (bbox → top 10 species)               (species → top 10 locations    │
│                                              + full 2 409-cell heatmap)    │
│                                                                            │
│                               FastAPI  ◄────────────►  Leaflet single-page │
└───────────────────────────────────────────────────────────────────────────┘
```

## Two query modes

1. **Bounding box → top species.** The user drags a rectangle on the map (or
   types lat/lon bounds). The backend scores every grid cell inside the bbox
   for each of the next 7 days with all 369 classifiers and returns the
   **top 10 most likely fish species** (ranked by lift = bbox-mean probability
   minus grid-wide-mean probability). A heat map of the top-1 species is
   painted over the bbox.
2. **Species → top locations.** The user picks a species from the 369-item
   autocompleting dropdown. The backend returns the **top 10 grid cells**
   ranked by 7-day mean probability, along with the full 2 409-cell
   probability heatmap. Cells with no training observations are dimmed.

## Data pipeline

### Model 1 — real-time ocean feature forecaster

A 30-day rolling cache of real, daily ocean state on the 0.25° × 0.25°
US-west-coast mesh (30–48 °N, −124 to −116 °W, 73 × 33 = 2 409 cells).

| Variable            | Source                                                           | Auth |
|---------------------|------------------------------------------------------------------|------|
| SST, SST anomaly    | NOAA OISST v2.1 NRT (`ncdcOisst21NrtAgg_LonPM180`) via ERDDAP    | none |
| Chlorophyll-a       | VIIRS DINEOF gap-filled daily (`noaacwNPPN20VIIRSDINEOFDaily`)   | none |
| SSH (sea-level)     | CMEMS DUACS NRT L4 (`cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D`) | CMEMS |
| Salinity (surface)  | GLORYS operational (`cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m`)  | CMEMS |
| Dissolved O₂        | CMEMS BGC operational (`cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m`) | CMEMS |

* On first boot the server downloads 30 days of real data (≈ 10 min).
* On each refresh only the **single newest day** is fetched; the oldest day
  is evicted so the window stays exactly 30 days.
* SST anomaly uses a DOY climatology computed from the OISST final product
  (2018–2024 reference decade), persisted on disk.
* The 7-day forecast blends a per-cell exponential-weighted moving average
  (alpha = 0.35) of the 30-day window with the 30-day mean that serves as a
  local climatology. Blend weight decays from 0.80 at day +1 to ~0.45 at
  day +7.

**No synthetic data.** If a feed fails the backend raises and the
associated query returns an HTTP 502 with the upstream error.

### Model 2 — species distribution classifier

`peter_code` branch, `new_species_model/species_distribution_model.joblib`.

* 369 logistic-regression pipelines (`StandardScaler` + `LogisticRegression`,
  `class_weight='balanced'`, `max_iter=2000`) produced by
  `build_species_distribution_model.py`.
* 13-column feature vector per cell per day:
  `lat, lon, sst, sst_anomaly, chlorophyll_log1p, salinity, dissolved_oxygen,
   ssh, day_of_year_sin, day_of_year_cos, month_sin, month_cos, year` —
  identical to training.
* Training data came from CalCOFI larvae + iNaturalist + NOAA WCGBT.
* Per-cell observation counts from `species_probability_maps.csv` are
  surfaced in every response as `n_obs_in_cell`; the UI dims cells with
  zero observations.

## Project layout

```
webapp/
├── backend/
│   ├── requirements.txt
│   ├── run.sh                           # convenience launcher
│   └── app/
│       ├── main.py                      # FastAPI entrypoint
│       ├── config.py                    # grid + dataset IDs + credentials
│       ├── schemas.py                   # Pydantic models
│       ├── ml/
│       │   ├── model_loader.py          # loads species_distribution_model.joblib
│       │   ├── data_sources.py          # ERDDAP + CMEMS fetchers (real data only)
│       │   ├── ocean_forecaster.py      # 30-day rolling store + 7-day forecast
│       │   ├── feature_builder.py       # assembles Model-2 feature vectors
│       │   └── species_pipeline.py      # orchestrates Model1 → Model2
│       └── data/                        # runtime cache (gitignored)
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── README.md
```

## Setup

```bash
cd webapp/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export CMEMS_USERNAME="your_cmems_email"
export CMEMS_PASSWORD="your_cmems_password"
./run.sh                                  # → http://localhost:8000
```

Open `http://localhost:8000/` in a browser. The health indicator in the
top-right shows bootstrap progress (`downloading: 12/30 days`) while the
initial feature store is built. Once it turns green everything is ready.

## API

| Method | Path                   | Purpose                                           |
|--------|------------------------|---------------------------------------------------|
| GET    | `/api/health`          | Model + feature store status                      |
| GET    | `/api/bootstrap`       | Detailed download progress                        |
| GET    | `/api/grid`            | Static grid metadata (lat/lon ranges, cell count) |
| GET    | `/api/species`         | All 369 species (snake_case id + display name)    |
| POST   | `/api/query/bbox`      | Bounding box → top 10 species + heatmap           |
| POST   | `/api/query/species`   | Species → top 10 locations + full 2 409-cell heat |
| POST   | `/api/refresh`         | Manually pull the newest day and re-run Model 2   |
