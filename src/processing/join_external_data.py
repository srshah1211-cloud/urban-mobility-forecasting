from __future__ import annotations

from pathlib import Path

import pandas as pd

WEATHER_COLUMNS = [
    "datetime_hour",
    "temperature",
    "humidity",
    "precipitation",
    "rain",
    "wind_speed",
    "is_raining",
]


def load_weather(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path, parse_dates=["datetime_hour"])
    missing = set(WEATHER_COLUMNS) - set(weather.columns)
    if missing:
        raise ValueError(f"Weather data is missing columns: {sorted(missing)}")
    return weather[WEATHER_COLUMNS]


def load_holidays(path: Path) -> pd.DataFrame:
    holiday_data = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "holiday_name", "is_holiday"}
    missing = required - set(holiday_data.columns)
    if missing:
        raise ValueError(f"Holiday data is missing columns: {sorted(missing)}")
    holiday_data["date"] = holiday_data["date"].dt.date
    return holiday_data[["date", "holiday_name", "is_holiday"]]


def join_external_data(
    demand: pd.DataFrame,
    zones: pd.DataFrame,
    weather: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    features = demand.merge(
        zones,
        left_on="zone_id",
        right_on="location_id",
        how="left",
    ).drop(columns=["location_id"])
    features = features.merge(weather, on="datetime_hour", how="left")

    features["date"] = features["datetime_hour"].dt.date
    features = features.merge(holidays, on="date", how="left")
    features["is_holiday"] = features["is_holiday"].fillna(False).astype(bool)
    features["holiday_name"] = features["holiday_name"].fillna("")
    return features.drop(columns=["date"])
