from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.config import paths, settings

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "rain",
    "precipitation",
    "relative_humidity_2m",
]


def fetch_open_meteo_weather(
    start_date: str,
    end_date: str,
    latitude: float = settings.weather_latitude,
    longitude: float = settings.weather_longitude,
    timezone: str = settings.weather_timezone,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone,
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not hourly:
        raise ValueError(f"Open-Meteo response did not include hourly data: {payload}")

    weather = pd.DataFrame(hourly)
    weather = weather.rename(
        columns={
            "time": "datetime_hour",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
        }
    )
    weather["datetime_hour"] = pd.to_datetime(weather["datetime_hour"])
    weather["is_raining"] = weather["precipitation"].fillna(0) > 0
    return weather[
        [
            "datetime_hour",
            "temperature",
            "humidity",
            "precipitation",
            "rain",
            "wind_speed",
            "is_raining",
        ]
    ]


def write_weather_csv(
    start_date: str,
    end_date: str,
    output_path: Path | None = None,
) -> Path:
    if output_path is None:
        label = f"{start_date.replace('-', '_')}_{end_date.replace('-', '_')}"
        output_path = paths.external_data / f"weather_nyc_{label}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather = fetch_open_meteo_weather(start_date, end_date)
    weather.to_csv(output_path, index=False)
    return output_path


def load_weather_csv(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path, parse_dates=["datetime_hour"])
    required = [
        "datetime_hour",
        "temperature",
        "humidity",
        "precipitation",
        "rain",
        "wind_speed",
        "is_raining",
    ]
    missing = set(required) - set(weather.columns)
    if missing:
        raise ValueError(f"Weather CSV is missing columns: {sorted(missing)}")
    return weather[required]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch hourly NYC weather data.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = write_weather_csv(args.start_date, args.end_date, args.output)
    print(output)


if __name__ == "__main__":
    main()
