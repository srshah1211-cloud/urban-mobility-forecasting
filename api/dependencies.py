from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import paths
from src.database.read import (
    DatabaseReadError,
    read_feature_table,
    read_hourly_demand,
    read_taxi_zones,
)
from src.ingestion.load_zone_lookup import load_zone_lookup


@lru_cache(maxsize=1)
def get_zones() -> pd.DataFrame:
    try:
        zones = read_taxi_zones()
        if not zones.empty:
            return zones
    except DatabaseReadError:
        pass
    return load_zone_lookup()


@lru_cache(maxsize=1)
def get_hourly_demand() -> pd.DataFrame:
    try:
        demand = read_hourly_demand()
        if not demand.empty:
            demand["datetime_hour"] = pd.to_datetime(demand["datetime_hour"])
            return demand
    except DatabaseReadError:
        pass
    return pd.read_parquet(paths.hourly_demand)


@lru_cache(maxsize=1)
def get_feature_table() -> pd.DataFrame:
    try:
        features = read_feature_table()
        if not features.empty:
            features["datetime_hour"] = pd.to_datetime(features["datetime_hour"])
            return features
    except DatabaseReadError:
        pass
    return pd.read_parquet(paths.feature_table)
