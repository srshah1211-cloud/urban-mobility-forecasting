from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.config import paths

REQUIRED_TAXI_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "total_amount",
]


def list_taxi_files(raw_dir: Path = paths.raw_data) -> list[Path]:
    return sorted(raw_dir.glob("yellow_tripdata_*.parquet"))


def read_taxi_file(
    path: Path, columns: Iterable[str] = REQUIRED_TAXI_COLUMNS
) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns))


def read_taxi_files(files: Iterable[Path]) -> pd.DataFrame:
    frames = [read_taxi_file(path) for path in files]
    if not frames:
        raise FileNotFoundError("No yellow_tripdata_*.parquet files were found.")
    return pd.concat(frames, ignore_index=True)
