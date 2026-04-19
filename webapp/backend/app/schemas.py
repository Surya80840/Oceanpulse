"""Pydantic request / response models for the OceanPulse API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BBoxQuery(BaseModel):
    lat_min: float = Field(..., ge=30.0, le=48.0, description="Southern bound (deg N).")
    lat_max: float = Field(..., ge=30.0, le=48.0, description="Northern bound (deg N).")
    lon_min: float = Field(..., ge=-124.0, le=-116.0, description="Western bound (deg E, negative).")
    lon_max: float = Field(..., ge=-124.0, le=-116.0, description="Eastern bound (deg E, negative).")
    top_k: int = Field(10, ge=1, le=50)


class SpeciesQuery(BaseModel):
    species: str
    top_k: int = Field(10, ge=1, le=100)


class CellProbability(BaseModel):
    lat: float
    lon: float
    probability: float
    n_obs_in_cell: int = 0


class SpeciesScore(BaseModel):
    species: str
    display_name: str
    mean_probability: float
    max_probability: float
    n_cells: int


class BBoxResponse(BaseModel):
    forecast_start: str
    forecast_end: str
    n_cells: int
    top_species: List[SpeciesScore]
    heatmap: List[CellProbability]          # heatmap of the top-1 species inside the bbox
    heatmap_species: Optional[str] = None


class SpeciesResponse(BaseModel):
    species: str
    display_name: str
    forecast_start: str
    forecast_end: str
    top_locations: List[CellProbability]
    heatmap: List[CellProbability]          # every grid cell


class GridInfo(BaseModel):
    lat_min: float
    lat_max: float
    lat_step: float
    lon_min: float
    lon_max: float
    lon_step: float
    n_lat: int
    n_lon: int
    n_cells: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_species: int
    feature_store_days: int
    last_refresh: Optional[str] = None
    live_feeds: bool
