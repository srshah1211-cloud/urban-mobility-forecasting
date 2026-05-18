from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_engine


class DatabaseReadError(RuntimeError):
    pass


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    try:
        with get_engine().connect() as connection:
            return pd.read_sql_query(text(query), connection, params=params)
    except SQLAlchemyError as exc:
        raise DatabaseReadError(str(exc)) from exc


def read_taxi_zones() -> pd.DataFrame:
    return read_sql(
        """
        SELECT
            location_id,
            borough,
            zone,
            service_zone
        FROM taxi_zones
        ORDER BY borough, zone
        """
    )


def read_hourly_demand() -> pd.DataFrame:
    return read_sql(
        """
        SELECT
            datetime_hour,
            zone_id,
            trip_count,
            avg_trip_distance,
            avg_fare,
            avg_total_amount,
            avg_passenger_count
        FROM hourly_zone_demand
        ORDER BY datetime_hour, zone_id
        """
    )


def read_feature_table() -> pd.DataFrame:
    return read_sql(
        """
        SELECT
            datetime_hour,
            zone_id,
            borough,
            zone_name AS zone,
            trip_count,
            temperature,
            humidity,
            precipitation,
            rain,
            wind_speed,
            is_raining,
            hour,
            day_of_week,
            month,
            is_weekend,
            is_holiday,
            is_peak_hour,
            lag_1_hour_demand,
            lag_24_hour_demand,
            rolling_3_hour_avg,
            rolling_24_hour_avg
        FROM feature_table
        ORDER BY datetime_hour, zone_id
        """
    )


def read_model_runs() -> pd.DataFrame:
    return read_sql(
        """
        SELECT
            model_run_id,
            model_name,
            model_version,
            train_start_date,
            train_end_date,
            test_start_date,
            test_end_date,
            mae,
            rmse,
            mape,
            r2,
            model_path,
            created_at
        FROM model_runs
        ORDER BY created_at DESC, model_run_id DESC
        """
    )


def read_model_predictions() -> pd.DataFrame:
    return read_sql(
        """
        WITH latest_run AS (
            SELECT model_run_id
            FROM model_runs
            WHERE model_path IS NOT NULL
            ORDER BY created_at DESC, model_run_id DESC
            LIMIT 1
        )
        SELECT
            p.model_run_id,
            r.model_name,
            r.model_version,
            p.datetime_hour,
            p.zone_id,
            z.zone,
            z.borough,
            p.actual_trip_count,
            p.predicted_trip_count,
            p.prediction_created_at
        FROM model_predictions p
        JOIN latest_run lr ON p.model_run_id = lr.model_run_id
        JOIN model_runs r ON p.model_run_id = r.model_run_id
        LEFT JOIN taxi_zones z ON p.zone_id = z.location_id
        ORDER BY p.datetime_hour, p.zone_id
        """
    )
