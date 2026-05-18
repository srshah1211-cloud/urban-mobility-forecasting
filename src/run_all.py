from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import paths
from src.database.connection import get_engine
from src.database.load_to_postgres import load_processed_tables, load_static_tables
from src.database.schema import apply_schema
from src.ingestion.load_holidays import write_us_holidays
from src.ingestion.load_weather_data import write_weather_csv
from src.models.train import train_models
from src.pipeline import build_processed_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def default_weather_path(start_date: str, end_date: str) -> Path:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    label = f"{start.year}_{start.month:02}_{end.month:02}"
    return paths.external_data / f"weather_nyc_{label}.csv"


def default_holidays_path(start_date: str, end_date: str) -> Path:
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    label = str(start_year) if start_year == end_year else f"{start_year}_{end_year}"
    return paths.external_data / f"us_holidays_{label}.csv"


def year_range(start_date: str, end_date: str) -> list[int]:
    return list(range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1))


def run_all(
    *,
    start_date: str,
    end_date: str,
    weather_path: Path | None = None,
    holidays_path: Path | None = None,
    fetch_weather: bool = True,
    replace: bool = True,
    train_after_load: bool = False,
    future_days: int = 30,
) -> None:
    paths.external_data.mkdir(parents=True, exist_ok=True)
    paths.processed_data.mkdir(parents=True, exist_ok=True)

    weather_path = weather_path or default_weather_path(start_date, end_date)
    holidays_path = holidays_path or default_holidays_path(start_date, end_date)

    logger.info("Generating holidays at %s", holidays_path)
    write_us_holidays(year_range(start_date, end_date), holidays_path)

    if fetch_weather:
        logger.info("Fetching weather at %s", weather_path)
        write_weather_csv(start_date, end_date, weather_path)
    elif not weather_path.exists():
        raise FileNotFoundError(
            f"Weather file not found: {weather_path}. Remove --use-existing-weather "
            "to fetch it from Open-Meteo."
        )

    logger.info("Using weather file %s", weather_path)

    logger.info("Building processed parquet datasets")
    demand, features = build_processed_data(
        start_date=start_date,
        end_date=end_date,
        weather_path=weather_path,
        holidays_path=holidays_path,
    )
    logger.info("Built %s demand rows and %s feature rows", len(demand), len(features))

    logger.info("Applying Postgres schema")
    apply_schema()

    logger.info("Loading processed data into Postgres")
    engine = get_engine()
    load_static_tables(
        engine,
        weather_path=weather_path,
        holidays_path=holidays_path,
        replace=replace,
    )
    load_processed_tables(engine, replace=False)

    if train_after_load:
        logger.info("Training models and storing predictions in Postgres")
        train_models(
            paths.feature_table,
            store_to_postgres=True,
            future_days=future_days,
        )

    logger.info("Finished full ingestion, processing, and Postgres load")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate external data, build processed datasets, and load Postgres."
        )
    )
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-03-31")
    parser.add_argument("--weather-path", type=Path)
    parser.add_argument("--holidays-path", type=Path)
    parser.add_argument(
        "--use-existing-weather",
        action="store_true",
        help="Use the weather CSV on disk instead of fetching from Open-Meteo.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing Postgres rows instead of replacing loaded tables.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models after loading Postgres and store predictions.",
    )
    parser.add_argument("--future-days", type=int, default=30)
    args = parser.parse_args()

    run_all(
        start_date=args.start_date,
        end_date=args.end_date,
        weather_path=args.weather_path,
        holidays_path=args.holidays_path,
        fetch_weather=not args.use_existing_weather,
        replace=not args.append,
        train_after_load=args.train,
        future_days=args.future_days,
    )


if __name__ == "__main__":
    main()
