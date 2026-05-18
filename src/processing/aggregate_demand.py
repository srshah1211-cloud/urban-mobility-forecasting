from __future__ import annotations

import pandas as pd


def aggregate_hourly_zone_demand(
    trips: pd.DataFrame,
    zone_ids: list[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    include_zero_demand: bool = True,
) -> pd.DataFrame:
    demand = (
        trips.groupby(["datetime_hour", "PULocationID"], as_index=False)
        .agg(
            trip_count=("PULocationID", "size"),
            avg_trip_distance=("trip_distance", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_total_amount=("total_amount", "mean"),
            avg_passenger_count=("passenger_count", "mean"),
        )
        .rename(columns={"PULocationID": "zone_id"})
    )

    if not include_zero_demand:
        return demand.sort_values(["datetime_hour", "zone_id"]).reset_index(drop=True)

    if zone_ids is None:
        zone_ids = sorted(demand["zone_id"].dropna().astype("int64").unique().tolist())

    min_hour = pd.Timestamp(start_date) if start_date else demand["datetime_hour"].min()
    max_hour = (
        pd.Timestamp(end_date) + pd.Timedelta(hours=23)
        if end_date
        else demand["datetime_hour"].max()
    )
    all_hours = pd.date_range(min_hour, max_hour, freq="h")
    full_index = pd.MultiIndex.from_product(
        [all_hours, sorted(zone_ids)], names=["datetime_hour", "zone_id"]
    ).to_frame(index=False)

    demand = full_index.merge(demand, on=["datetime_hour", "zone_id"], how="left")
    demand["trip_count"] = demand["trip_count"].fillna(0).astype("int64")
    avg_columns = [
        "avg_trip_distance",
        "avg_fare",
        "avg_total_amount",
        "avg_passenger_count",
    ]
    demand[avg_columns] = demand[avg_columns].fillna(0.0)
    return demand.sort_values(["datetime_hour", "zone_id"]).reset_index(drop=True)
