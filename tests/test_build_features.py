from __future__ import annotations

import pandas as pd

from src.features.build_features import build_feature_table


def test_build_feature_table_adds_calendar_and_history_by_zone() -> None:
    frame = pd.DataFrame(
        {
            "datetime_hour": pd.date_range(
                "2026-01-01 07:00:00", periods=4, freq="h"
            ).tolist()
            * 2,
            "zone_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "trip_count": [10, 20, 30, 40, 1, 2, 3, 4],
            "is_holiday": [True] * 8,
            "is_raining": [False] * 8,
        }
    )

    features = build_feature_table(frame)

    zone_2_second_hour = features[
        (features["zone_id"] == 2)
        & (features["datetime_hour"] == pd.Timestamp("2026-01-01 08:00:00"))
    ].iloc[0]
    assert zone_2_second_hour["hour"] == 8
    assert bool(zone_2_second_hour["is_peak_hour"]) is True
    assert zone_2_second_hour["lag_1_hour_demand"] == 1
    assert zone_2_second_hour["rolling_3_hour_avg"] == 1
