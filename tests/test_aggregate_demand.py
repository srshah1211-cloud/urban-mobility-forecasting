from __future__ import annotations

import pandas as pd

from src.processing.aggregate_demand import aggregate_hourly_zone_demand


def test_aggregate_hourly_zone_demand_includes_zero_rows() -> None:
    trips = pd.DataFrame(
        {
            "datetime_hour": [pd.Timestamp("2026-01-01 08:00:00")],
            "PULocationID": [161],
            "trip_distance": [1.5],
            "fare_amount": [12.0],
            "total_amount": [15.0],
            "passenger_count": [1.0],
        }
    )

    demand = aggregate_hourly_zone_demand(
        trips,
        zone_ids=[161, 162],
        start_date="2026-01-01",
        end_date="2026-01-01",
    )

    assert len(demand) == 48
    actual = demand[
        (demand["datetime_hour"] == pd.Timestamp("2026-01-01 08:00:00"))
        & (demand["zone_id"] == 161)
    ].iloc[0]
    zero = demand[
        (demand["datetime_hour"] == pd.Timestamp("2026-01-01 08:00:00"))
        & (demand["zone_id"] == 162)
    ].iloc[0]
    assert actual["trip_count"] == 1
    assert zero["trip_count"] == 0
