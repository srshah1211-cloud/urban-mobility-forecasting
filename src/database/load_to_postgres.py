from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import paths
from src.database.connection import get_engine
from src.ingestion.load_zone_lookup import load_zone_lookup
from src.processing.join_external_data import load_holidays, load_weather


def load_dataframe(
    frame: pd.DataFrame,
    table_name: str,
    engine: Engine,
    if_exists: str = "append",
) -> None:
    frame.to_sql(table_name, engine, if_exists=if_exists, index=False, method="multi")


def clear_loaded_tables(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                TRUNCATE TABLE
                    model_predictions,
                    model_runs,
                    feature_table,
                    hourly_zone_demand,
                    holidays,
                    weather_hourly,
                    taxi_zones
                RESTART IDENTITY CASCADE
                """
            )
        )


def load_static_tables(
    engine: Engine,
    weather_path: Path,
    holidays_path: Path,
    replace: bool = True,
) -> None:
    if replace:
        clear_loaded_tables(engine)
    load_dataframe(load_zone_lookup(), "taxi_zones", engine)
    load_dataframe(load_weather(weather_path), "weather_hourly", engine)
    load_dataframe(load_holidays(holidays_path), "holidays", engine)


def load_processed_tables(engine: Engine, replace: bool = True) -> None:
    demand_columns = [
        "datetime_hour",
        "zone_id",
        "trip_count",
        "avg_trip_distance",
        "avg_fare",
        "avg_total_amount",
        "avg_passenger_count",
    ]
    feature_columns = [
        "datetime_hour",
        "zone_id",
        "borough",
        "zone_name",
        "trip_count",
        "temperature",
        "humidity",
        "precipitation",
        "rain",
        "wind_speed",
        "is_raining",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "is_peak_hour",
        "lag_1_hour_demand",
        "lag_24_hour_demand",
        "rolling_3_hour_avg",
        "rolling_24_hour_avg",
    ]
    if replace:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    TRUNCATE TABLE
                        model_predictions,
                        model_runs,
                        feature_table,
                        hourly_zone_demand
                    RESTART IDENTITY CASCADE
                    """
                )
            )
    demand = pd.read_parquet(paths.hourly_demand)[demand_columns]
    load_dataframe(demand, "hourly_zone_demand", engine)

    feature_table = pd.read_parquet(paths.feature_table).rename(
        columns={"zone": "zone_name"}
    )
    load_dataframe(feature_table[feature_columns], "feature_table", engine)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load local CSV/parquet data into Postgres."
    )
    parser.add_argument("--weather-path", type=Path, required=True)
    parser.add_argument("--holidays-path", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    engine = get_engine()
    load_static_tables(
        engine, args.weather_path, args.holidays_path, replace=not args.append
    )
    load_processed_tables(engine, replace=not args.append)


if __name__ == "__main__":
    main()
