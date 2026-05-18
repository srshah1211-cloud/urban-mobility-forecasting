from __future__ import annotations

import pandas as pd


def clean_taxi_trips(
    trips: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    trips = trips.copy()
    trips["tpep_pickup_datetime"] = pd.to_datetime(trips["tpep_pickup_datetime"])
    trips["tpep_dropoff_datetime"] = pd.to_datetime(trips["tpep_dropoff_datetime"])

    mask = (
        trips["PULocationID"].notna()
        & trips["tpep_pickup_datetime"].notna()
        & trips["tpep_dropoff_datetime"].notna()
        & (trips["tpep_dropoff_datetime"] > trips["tpep_pickup_datetime"])
        & (trips["trip_distance"].fillna(0) >= 0)
        & (trips["fare_amount"].fillna(0) >= 0)
        & (trips["total_amount"].fillna(0) >= 0)
        & (trips["passenger_count"].fillna(0) >= 0)
    )

    if start_date:
        mask &= trips["tpep_pickup_datetime"] >= pd.Timestamp(start_date)
    if end_date:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        mask &= trips["tpep_pickup_datetime"] < end_exclusive

    trips = trips.loc[mask].copy()
    trips["PULocationID"] = trips["PULocationID"].astype("int64")
    trips["datetime_hour"] = trips["tpep_pickup_datetime"].dt.floor("h")
    return trips
