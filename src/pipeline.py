from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import paths
from src.features.build_features import build_feature_table
from src.ingestion.load_holidays import write_us_holidays
from src.ingestion.load_taxi_data import list_taxi_files, read_taxi_files
from src.ingestion.load_weather_data import write_weather_csv
from src.ingestion.load_zone_lookup import load_zone_lookup
from src.processing.aggregate_demand import aggregate_hourly_zone_demand
from src.processing.clean_taxi_data import clean_taxi_trips
from src.processing.join_external_data import (
    join_external_data,
    load_holidays,
    load_weather,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_processed_data(
    start_date: str,
    end_date: str,
    weather_path: Path,
    holidays_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    taxi_files = list_taxi_files()
    logger.info("Reading %s taxi parquet files", len(taxi_files))
    trips = read_taxi_files(taxi_files)
    trips = clean_taxi_trips(trips, start_date=start_date, end_date=end_date)

    zones = load_zone_lookup()
    zone_ids = zones["location_id"].astype("int64").tolist()
    demand = aggregate_hourly_zone_demand(
        trips,
        zone_ids=zone_ids,
        start_date=start_date,
        end_date=end_date,
        include_zero_demand=True,
    )

    weather = load_weather(weather_path)
    holiday_data = load_holidays(holidays_path)
    feature_base = join_external_data(demand, zones, weather, holiday_data)
    features = build_feature_table(feature_base)

    paths.processed_data.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(paths.hourly_demand, index=False)
    features.to_parquet(paths.feature_table, index=False)
    return demand, features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local data processing pipeline."
    )
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--fetch-weather", action="store_true")
    parser.add_argument("--generate-holidays", action="store_true")
    parser.add_argument("--weather-path", type=Path)
    parser.add_argument("--holidays-path", type=Path)
    args = parser.parse_args()

    weather_path = args.weather_path
    if args.fetch_weather:
        weather_path = write_weather_csv(args.start_date, args.end_date, weather_path)
    if weather_path is None:
        label = f"{args.start_date.replace('-', '_')}_{args.end_date.replace('-', '_')}"
        weather_path = paths.external_data / f"weather_nyc_{label}.csv"

    holidays_path = args.holidays_path
    if args.generate_holidays:
        year_range = range(
            pd.Timestamp(args.start_date).year, pd.Timestamp(args.end_date).year + 1
        )
        holidays_path = write_us_holidays(list(year_range), holidays_path)
    if holidays_path is None:
        holidays_path = (
            paths.external_data
            / f"us_holidays_{pd.Timestamp(args.start_date).year}.csv"
        )

    demand, features = build_processed_data(
        args.start_date,
        args.end_date,
        weather_path=weather_path,
        holidays_path=holidays_path,
    )
    logger.info("Wrote %s rows to %s", len(demand), paths.hourly_demand)
    logger.info("Wrote %s rows to %s", len(features), paths.feature_table)


if __name__ == "__main__":
    main()
