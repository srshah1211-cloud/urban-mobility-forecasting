from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import paths
from src.database.read import DatabaseReadError, read_taxi_zones
from src.database.write import (
    DatabaseWriteError,
    insert_model_predictions,
    insert_model_run,
)
from src.ingestion.load_holidays import generate_us_holidays
from src.models.config import FEATURE_COLUMNS
from src.models.evaluate import regression_metrics

WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "precipitation",
    "rain",
    "wind_speed",
    "is_raining",
]


def train_test_split_by_time(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_hours = frame["datetime_hour"].drop_duplicates().sort_values()
    cutoff_position = max(1, int(len(ordered_hours) * (1 - test_fraction)))
    cutoff_hour = ordered_hours.iloc[cutoff_position]
    train = frame[frame["datetime_hour"] < cutoff_hour]
    test = frame[frame["datetime_hour"] >= cutoff_hour]
    return train, test


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = prepare_feature_frame(frame)
    y = frame["trip_count"]
    return x, y


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[FEATURE_COLUMNS].copy()
    bool_columns = x.select_dtypes(include=["bool"]).columns
    x[bool_columns] = x[bool_columns].astype("int8")
    x = x.apply(pd.to_numeric, errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_baseline(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[float, dict[str, float]]:
    means = train.groupby(["zone_id", "hour"])["trip_count"].mean()
    predictions = [
        means.get((row.zone_id, row.hour), train["trip_count"].mean())
        for row in test[["zone_id", "hour"]].itertuples(index=False)
    ]
    return train["trip_count"].mean(), regression_metrics(
        test["trip_count"], predictions
    )


def train_models(
    feature_path: Path = paths.feature_table,
    store_to_postgres: bool = True,
    future_days: int = 30,
) -> pd.DataFrame:
    features = pd.read_parquet(feature_path)
    train, test = train_test_split_by_time(features)
    x_train, y_train = prepare_xy(train)
    x_test, y_test = prepare_xy(test)

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        pass

    results = []
    _, baseline_metrics = train_baseline(train, test)
    results.append({"model_name": "historical_average", **baseline_metrics})

    paths.models.mkdir(parents=True, exist_ok=True)
    best_model = None
    best_name = ""
    best_mae = float("inf")
    best_predictions = None
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = np.maximum(model.predict(x_test), 0.0)
        metrics = regression_metrics(y_test, predictions)
        results.append({"model_name": model_name, **metrics})
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model = model
            best_name = model_name
            best_predictions = predictions

    best_model_path = None
    if best_model is not None:
        best_model_path = paths.models / f"{best_name}_model.pkl"
        joblib.dump(best_model, best_model_path)
        with (paths.models / "feature_columns.json").open("w") as file:
            json.dump(FEATURE_COLUMNS, file, indent=2)

    report = pd.DataFrame(results).sort_values("mae")
    report.to_csv(paths.models / "model_metrics.csv", index=False)

    if store_to_postgres and best_model is not None and best_predictions is not None:
        try:
            store_training_outputs(
                report=report,
                best_model_name=best_name,
                best_model_path=best_model_path,
                train=train,
                test=test,
                test_predictions=best_predictions,
                features=features,
                model=best_model,
                future_days=future_days,
            )
        except DatabaseWriteError as exc:
            print(f"Skipped Postgres model storage: {exc}")

    return report


def store_training_outputs(
    *,
    report: pd.DataFrame,
    best_model_name: str,
    best_model_path: Path | None,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_predictions: np.ndarray,
    features: pd.DataFrame,
    model,
    future_days: int,
) -> None:
    ensure_postgres_prediction_prerequisites()

    model_version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    train_start = train["datetime_hour"].min().date()
    train_end = train["datetime_hour"].max().date()
    test_start = test["datetime_hour"].min().date()
    test_end = test["datetime_hour"].max().date()

    best_model_run_id = None
    for row in report.itertuples(index=False):
        is_best = row.model_name == best_model_name
        model_run_id = insert_model_run(
            model_name=row.model_name,
            model_version=model_version,
            train_start_date=train_start,
            train_end_date=train_end,
            test_start_date=test_start,
            test_end_date=test_end,
            mae=float(row.mae),
            rmse=float(row.rmse),
            mape=float(row.mape),
            r2=float(row.r2),
            model_path=str(best_model_path) if is_best and best_model_path else None,
        )
        if is_best:
            best_model_run_id = model_run_id

    if best_model_run_id is None:
        return

    holdout = build_holdout_prediction_frame(
        model_run_id=best_model_run_id,
        test=test,
        predictions=test_predictions,
    )
    future = build_future_prediction_frame(
        model_run_id=best_model_run_id,
        features=features,
        model=model,
        future_days=future_days,
    )
    insert_model_predictions(pd.concat([holdout, future], ignore_index=True))


def ensure_postgres_prediction_prerequisites() -> None:
    try:
        zones = read_taxi_zones()
    except DatabaseReadError as exc:
        raise DatabaseWriteError(str(exc)) from exc

    if zones.empty:
        raise DatabaseWriteError(
            "taxi_zones is empty. Run `python3 -m src.database.load_to_postgres` "
            "before training with Postgres storage."
        )


def build_holdout_prediction_frame(
    *,
    model_run_id: int,
    test: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_run_id": model_run_id,
            "datetime_hour": pd.to_datetime(test["datetime_hour"]).to_numpy(),
            "zone_id": test["zone_id"].astype("int64").to_numpy(),
            "actual_trip_count": test["trip_count"].astype("int64").to_numpy(),
            "predicted_trip_count": predictions,
        }
    )


def build_future_prediction_frame(
    *,
    model_run_id: int,
    features: pd.DataFrame,
    model,
    future_days: int,
) -> pd.DataFrame:
    if future_days <= 0:
        return pd.DataFrame(
            columns=[
                "model_run_id",
                "datetime_hour",
                "zone_id",
                "actual_trip_count",
                "predicted_trip_count",
            ]
        )

    features = features.sort_values(["zone_id", "datetime_hour"]).copy()
    future_hours = pd.date_range(
        features["datetime_hour"].max() + pd.Timedelta(hours=1),
        periods=future_days * 24,
        freq="h",
    )
    weather_profiles = build_weather_profiles(features)
    holiday_dates = set(
        generate_us_holidays(sorted({hour.year for hour in future_hours}))["date"]
    )
    histories = {
        zone_id: group["trip_count"].tail(24).astype(float).tolist()
        for zone_id, group in features.groupby("zone_id")
    }

    rows = []
    for prediction_hour in future_hours:
        batch_rows = []
        for zone_id, history in histories.items():
            weather = estimate_weather(weather_profiles, prediction_hour)
            batch_rows.append(
                {
                    "zone_id": int(zone_id),
                    "hour": prediction_hour.hour,
                    "day_of_week": prediction_hour.dayofweek,
                    "month": prediction_hour.month,
                    "is_weekend": prediction_hour.dayofweek in {5, 6},
                    "is_holiday": prediction_hour.date() in holiday_dates,
                    "is_peak_hour": prediction_hour.hour in {7, 8, 9, 17, 18, 19},
                    "lag_1_hour_demand": history[-1] if history else 0.0,
                    "lag_24_hour_demand": history[-24] if len(history) >= 24 else 0.0,
                    "rolling_3_hour_avg": (
                        float(np.mean(history[-3:])) if history else 0.0
                    ),
                    "rolling_24_hour_avg": (
                        float(np.mean(history[-24:])) if history else 0.0
                    ),
                    **weather,
                }
            )

        x_future = prepare_feature_frame(pd.DataFrame(batch_rows))
        predictions = np.maximum(model.predict(x_future), 0.0)
        for row, prediction in zip(batch_rows, predictions, strict=False):
            zone_id = row["zone_id"]
            rows.append(
                {
                    "model_run_id": model_run_id,
                    "datetime_hour": prediction_hour,
                    "zone_id": zone_id,
                    "actual_trip_count": None,
                    "predicted_trip_count": float(prediction),
                }
            )
            histories[zone_id].append(float(prediction))

    return pd.DataFrame(rows)


def build_weather_profiles(
    features: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    numeric_weather = features.copy()
    numeric_weather["is_raining"] = numeric_weather["is_raining"].astype("int8")
    return {
        "month_hour": numeric_weather.groupby(["month", "hour"])[
            WEATHER_COLUMNS
        ].mean(),
        "hour": numeric_weather.groupby("hour")[WEATHER_COLUMNS].mean(),
        "global": numeric_weather[WEATHER_COLUMNS].mean(),
    }


def estimate_weather(
    weather_profiles: dict[str, pd.DataFrame | pd.Series],
    prediction_hour: pd.Timestamp,
) -> dict[str, float | bool]:
    month_hour = weather_profiles["month_hour"]
    hour = weather_profiles["hour"]
    if (prediction_hour.month, prediction_hour.hour) in month_hour.index:
        weather = month_hour.loc[(prediction_hour.month, prediction_hour.hour)]
    elif prediction_hour.hour in hour.index:
        weather = hour.loc[prediction_hour.hour]
    else:
        weather = weather_profiles["global"]

    return {
        "temperature": float(weather["temperature"]),
        "humidity": float(weather["humidity"]),
        "precipitation": float(weather["precipitation"]),
        "rain": float(weather["rain"]),
        "wind_speed": float(weather["wind_speed"]),
        "is_raining": bool(round(float(weather["is_raining"]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train taxi demand forecasting models."
    )
    parser.add_argument("--feature-path", type=Path, default=paths.feature_table)
    parser.add_argument("--future-days", type=int, default=30)
    parser.add_argument(
        "--no-store-to-postgres",
        action="store_false",
        dest="store_to_postgres",
        help="Train locally without writing model runs or predictions to Postgres.",
    )
    args = parser.parse_args()
    print(
        train_models(
            args.feature_path,
            store_to_postgres=args.store_to_postgres,
            future_days=args.future_days,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
