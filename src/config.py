from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    external_data: Path = PROJECT_ROOT / "data" / "external"
    processed_data: Path = PROJECT_ROOT / "data" / "processed"
    models: Path = PROJECT_ROOT / "models"
    zone_lookup: Path = PROJECT_ROOT / "data" / "external" / "taxi_zone_lookup.csv"
    hourly_demand: Path = (
        PROJECT_ROOT / "data" / "processed" / "hourly_zone_demand.parquet"
    )
    feature_table: Path = PROJECT_ROOT / "data" / "processed" / "feature_table.parquet"


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://urban_user:urban_password@localhost:5432/urban_mobility_db",
    )
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/xgboost_model.pkl"))
    feature_columns_path: Path = Path(
        os.getenv("FEATURE_COLUMNS_PATH", "models/feature_columns.json")
    )
    weather_latitude: float = 40.7128
    weather_longitude: float = -74.0060
    weather_timezone: str = "America/New_York"


paths = Paths()
settings = Settings()
