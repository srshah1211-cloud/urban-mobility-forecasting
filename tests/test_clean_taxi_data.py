from __future__ import annotations

import pandas as pd

from src.processing.clean_taxi_data import clean_taxi_trips


def test_clean_taxi_trips_filters_invalid_rows_and_adds_hour() -> None:
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2026-01-01 08:15:00",
                "2026-01-01 09:00:00",
                "2025-12-31 23:59:00",
            ],
            "tpep_dropoff_datetime": [
                "2026-01-01 08:30:00",
                "2026-01-01 08:55:00",
                "2026-01-01 00:10:00",
            ],
            "PULocationID": [161, 162, 163],
            "trip_distance": [1.2, 1.0, 2.0],
            "fare_amount": [10.0, 8.0, 12.0],
            "total_amount": [12.0, 10.0, 14.0],
            "passenger_count": [1.0, 1.0, 2.0],
        }
    )

    cleaned = clean_taxi_trips(raw, start_date="2026-01-01", end_date="2026-01-31")

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["PULocationID"] == 161
    assert cleaned.iloc[0]["datetime_hour"] == pd.Timestamp("2026-01-01 08:00:00")
