"""FastAPI entrypoint for the OceanPulse web app."""
from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .ml.model_loader import SpeciesDistributionModel
from .ml.ocean_forecaster import OceanFeatureStore
from .ml.species_pipeline import SpeciesForecastPipeline
from .schemas import (
    BBoxQuery,
    BBoxResponse,
    GridInfo,
    HealthResponse,
    SpeciesQuery,
    SpeciesResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("oceanpulse")


state: dict = {
    "bootstrap_status": {"stage": "pending", "done": 0, "total": 0, "message": ""},
}


def _progress_cb(i: int, total: int, day):
    st = state["bootstrap_status"]
    st["stage"] = "downloading"
    st["done"] = i
    st["total"] = total
    st["message"] = f"fetched {day}"


def _bootstrap_worker():
    """Runs in a background thread on startup.

    Populates the 30-day feature store from real feeds, then warms the
    Model1 -> Model2 cache so the first query returns instantly.
    """
    try:
        state["bootstrap_status"]["stage"] = "downloading"
        state["store"].bootstrap(progress_cb=_progress_cb)
        state["bootstrap_status"]["stage"] = "predicting"
        state["bootstrap_status"]["message"] = "running Model 1 -> Model 2"
        state["pipeline"]._ensure_cache()
        state["bootstrap_status"]["stage"] = "ready"
        state["bootstrap_status"]["message"] = "ok"
        log.info("Bootstrap finished; backend is ready.")
    except Exception as exc:                                           # noqa: BLE001
        log.exception("Bootstrap failed")
        state["bootstrap_status"]["stage"] = "error"
        state["bootstrap_status"]["message"] = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting OceanPulse backend...")
    model = SpeciesDistributionModel.load()
    store = OceanFeatureStore()
    pipeline = SpeciesForecastPipeline(model, store)

    state["model"] = model
    state["store"] = store
    state["pipeline"] = pipeline
    log.info("Model loaded (%d species).  Kicking off bootstrap.", len(model.species_columns))

    # If the cache already has a full 30 days we can go live immediately.
    already_have = store.n_days()
    if already_have >= config.HISTORY_DAYS:
        state["bootstrap_status"]["stage"] = "ready"
        state["bootstrap_status"]["message"] = "cache hit"
        state["bootstrap_status"]["done"] = already_have
        state["bootstrap_status"]["total"] = config.HISTORY_DAYS
        log.info("Feature store already has %d days; warming prediction cache.", already_have)
        pipeline._ensure_cache()
    else:
        state["bootstrap_status"]["total"] = config.HISTORY_DAYS - already_have
        threading.Thread(target=_bootstrap_worker, daemon=True, name="bootstrap").start()

    yield
    log.info("Shutting down OceanPulse backend.")


app = FastAPI(
    title="OceanPulse API",
    description="Two-stage ocean + species distribution forecasting service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _require_ready():
    stage = state.get("bootstrap_status", {}).get("stage")
    if stage != "ready":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service warming up.",
                "bootstrap": state.get("bootstrap_status"),
            },
        )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model: SpeciesDistributionModel | None = state.get("model")
    store: OceanFeatureStore | None = state.get("store")
    last_refresh = None
    if config.LAST_REFRESH_FILE.exists():
        last_refresh = config.LAST_REFRESH_FILE.read_text().strip()
    return HealthResponse(
        status=state["bootstrap_status"]["stage"],
        model_loaded=model is not None,
        n_species=len(model.species_columns) if model else 0,
        feature_store_days=store.n_days() if store else 0,
        last_refresh=last_refresh,
        live_feeds=True,
    )


@app.get("/api/bootstrap")
def bootstrap_status() -> dict:
    """Progress of the initial 30-day ocean download."""
    bs = dict(state.get("bootstrap_status", {}))
    store = state.get("store")
    if store is not None:
        bs["days_in_store"] = store.n_days()
        latest = store.latest_day()
        bs["latest_day"] = str(latest) if latest else None
    return bs


@app.get("/api/grid", response_model=GridInfo)
def grid() -> GridInfo:
    return GridInfo(
        lat_min=config.LAT_MIN,
        lat_max=config.LAT_MAX,
        lat_step=config.LAT_STEP,
        lon_min=config.LON_MIN,
        lon_max=config.LON_MAX,
        lon_step=config.LON_STEP,
        n_lat=config.N_LAT,
        n_lon=config.N_LON,
        n_cells=config.N_CELLS,
    )


@app.get("/api/species")
def species_list() -> dict:
    pipeline: SpeciesForecastPipeline = state["pipeline"]
    return {"species": pipeline.species_list()}


@app.post("/api/query/bbox", response_model=BBoxResponse)
def query_bbox(payload: BBoxQuery) -> BBoxResponse:
    _require_ready()
    pipeline: SpeciesForecastPipeline = state["pipeline"]
    try:
        result = pipeline.query_bbox(
            lat_min=payload.lat_min,
            lat_max=payload.lat_max,
            lon_min=payload.lon_min,
            lon_max=payload.lon_max,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return BBoxResponse(**result)


@app.post("/api/query/species", response_model=SpeciesResponse)
def query_species(payload: SpeciesQuery) -> SpeciesResponse:
    _require_ready()
    pipeline: SpeciesForecastPipeline = state["pipeline"]
    try:
        result = pipeline.query_species(payload.species, top_k=payload.top_k)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown species: {payload.species!r}",
        )
    return SpeciesResponse(**result)


@app.post("/api/refresh")
def refresh() -> dict:
    _require_ready()
    store: OceanFeatureStore = state["store"]
    pipeline: SpeciesForecastPipeline = state["pipeline"]
    try:
        result = store.refresh()
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {exc}")
    pipeline.invalidate()
    pipeline._ensure_cache()
    return result


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
frontend_dir = Path(config.FRONTEND_DIR)
if frontend_dir.exists():
    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.get("/app.js")
    def app_js() -> FileResponse:
        return FileResponse(frontend_dir / "app.js", media_type="application/javascript")

    @app.get("/styles.css")
    def app_css() -> FileResponse:
        return FileResponse(frontend_dir / "styles.css", media_type="text/css")

    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
