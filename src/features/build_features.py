from __future__ import annotations

import pandas as pd

PEAK_HOURS = {7, 8, 9, 17, 18, 19}


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    features["datetime_hour"] = pd.to_datetime(features["datetime_hour"])
    features["hour"] = features["datetime_hour"].dt.hour
    features["day_of_week"] = features["datetime_hour"].dt.dayofweek
    features["day_name"] = features["datetime_hour"].dt.day_name()
    features["month"] = features["datetime_hour"].dt.month
    features["is_weekend"] = features["day_of_week"].isin([5, 6])
    features["is_peak_hour"] = features["hour"].isin(PEAK_HOURS)
    return features


def add_demand_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.sort_values(["zone_id", "datetime_hour"]).copy()
    grouped = features.groupby("zone_id", group_keys=False)["trip_count"]
    features["lag_1_hour_demand"] = grouped.shift(1)
    features["lag_24_hour_demand"] = grouped.shift(24)
    features["rolling_3_hour_avg"] = grouped.transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    features["rolling_24_hour_avg"] = grouped.transform(
        lambda series: series.shift(1).rolling(24, min_periods=1).mean()
    )

    fill_columns = [
        "lag_1_hour_demand",
        "lag_24_hour_demand",
        "rolling_3_hour_avg",
        "rolling_24_hour_avg",
    ]
    features[fill_columns] = features[fill_columns].fillna(0.0)
    return features.sort_values(["datetime_hour", "zone_id"]).reset_index(drop=True)


def build_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    features = add_calendar_features(frame)
    features = add_demand_history_features(features)
    bool_columns = ["is_weekend", "is_holiday", "is_peak_hour", "is_raining"]
    for column in bool_columns:
        if column in features:
            features[column] = features[column].fillna(False).astype(bool)
    return features
