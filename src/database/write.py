from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_engine


class DatabaseWriteError(RuntimeError):
    pass


def insert_model_run(
    *,
    model_name: str,
    model_version: str,
    train_start_date,
    train_end_date,
    test_start_date,
    test_end_date,
    mae: float,
    rmse: float,
    mape: float,
    r2: float,
    model_path: str | None,
) -> int:
    query = text(
        """
        INSERT INTO model_runs (
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
            model_path
        )
        VALUES (
            :model_name,
            :model_version,
            :train_start_date,
            :train_end_date,
            :test_start_date,
            :test_end_date,
            :mae,
            :rmse,
            :mape,
            :r2,
            :model_path
        )
        RETURNING model_run_id
        """
    )
    params = {
        "model_name": model_name,
        "model_version": model_version,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "test_start_date": test_start_date,
        "test_end_date": test_end_date,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "model_path": model_path,
    }
    try:
        with get_engine().begin() as connection:
            return int(connection.execute(query, params).scalar_one())
    except SQLAlchemyError as exc:
        raise DatabaseWriteError(str(exc)) from exc


def insert_model_predictions(predictions: pd.DataFrame) -> None:
    columns = [
        "model_run_id",
        "datetime_hour",
        "zone_id",
        "actual_trip_count",
        "predicted_trip_count",
    ]
    try:
        predictions[columns].to_sql(
            "model_predictions",
            get_engine(),
            if_exists="append",
            index=False,
            chunksize=10_000,
            method="multi",
        )
    except SQLAlchemyError as exc:
        raise DatabaseWriteError(str(exc)) from exc
