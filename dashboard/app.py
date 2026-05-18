from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import paths  # noqa: E402
from src.database.read import (  # noqa: E402
    DatabaseReadError,
    read_feature_table,
    read_model_predictions,
    read_model_runs,
)  # noqa: E402
from src.models.config import FEATURE_COLUMNS  # noqa: E402
from src.models.registry import latest_model_path  # noqa: E402

st.set_page_config(page_title="Urban Mobility Forecasting", layout="wide")


@st.cache_data(show_spinner=False)
def load_features() -> tuple[pd.DataFrame, str]:
    try:
        features = read_feature_table()
        if not features.empty:
            features["datetime_hour"] = pd.to_datetime(features["datetime_hour"])
            return features, "Postgres"
    except DatabaseReadError:
        pass

    if not paths.feature_table.exists():
        raise FileNotFoundError(paths.feature_table)

    features = pd.read_parquet(paths.feature_table)
    features["datetime_hour"] = pd.to_datetime(features["datetime_hour"])
    return features, "Parquet fallback"


@st.cache_data(show_spinner=False)
def load_model_metrics() -> pd.DataFrame:
    try:
        runs = read_model_runs()
        if not runs.empty:
            latest_version = runs.iloc[0]["model_version"]
            return runs[runs["model_version"] == latest_version][
                ["model_name", "mae", "rmse", "mape", "r2", "created_at"]
            ]
    except DatabaseReadError:
        pass

    metrics_path = paths.models / "model_metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


@st.cache_data(show_spinner=False)
def load_model_predictions() -> pd.DataFrame:
    try:
        predictions = read_model_predictions()
        if not predictions.empty:
            predictions["datetime_hour"] = pd.to_datetime(predictions["datetime_hour"])
            return predictions
    except DatabaseReadError:
        pass
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str):
    return joblib.load(model_path)


def filter_features(
    features: pd.DataFrame,
    date_range: tuple,
    boroughs: list[str],
    zones: list[str],
) -> pd.DataFrame:
    start_date, end_date = date_range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    filtered = features[
        (features["datetime_hour"] >= start) & (features["datetime_hour"] < end)
    ]
    if boroughs:
        filtered = filtered[filtered["borough"].isin(boroughs)]
    if zones:
        filtered = filtered[filtered["zone"].isin(zones)]
    return filtered


def metric_row(features: pd.DataFrame) -> None:
    total_trips = int(features["trip_count"].sum())
    total_zones = int(features["zone_id"].nunique())
    avg_hourly_demand = features.groupby("datetime_hour")["trip_count"].sum().mean()
    peak_hour = (
        features.groupby("hour")["trip_count"].sum().idxmax()
        if not features.empty
        else 0
    )
    top_borough = (
        features.groupby("borough")["trip_count"].sum().idxmax()
        if not features.empty
        else "N/A"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total trips", f"{total_trips:,}")
    col2.metric("Zones", f"{total_zones:,}")
    col3.metric("Avg hourly demand", f"{avg_hourly_demand:,.0f}")
    col4.metric("Peak hour", f"{int(peak_hour):02d}:00")
    col5.metric("Top borough", top_borough)


def show_overview(features: pd.DataFrame) -> None:
    metric_row(features)
    hourly = features.groupby("datetime_hour", as_index=False)["trip_count"].sum()
    borough = (
        features.groupby("borough", as_index=False)["trip_count"]
        .sum()
        .sort_values("trip_count", ascending=False)
    )

    left, right = st.columns([2, 1])
    with left:
        fig = px.line(
            hourly,
            x="datetime_hour",
            y="trip_count",
            labels={"datetime_hour": "Time", "trip_count": "Trips"},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            borough,
            x="trip_count",
            y="borough",
            orientation="h",
            labels={"trip_count": "Trips", "borough": "Borough"},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


def show_trends(features: pd.DataFrame) -> None:
    daily = (
        features.assign(date=features["datetime_hour"].dt.date)
        .groupby("date", as_index=False)["trip_count"]
        .sum()
    )
    hour_pattern = features.groupby(["hour", "is_weekend"], as_index=False)[
        "trip_count"
    ].mean()
    hour_pattern["day_type"] = np.where(
        hour_pattern["is_weekend"], "Weekend", "Weekday"
    )

    left, right = st.columns(2)
    with left:
        fig = px.line(
            daily,
            x="date",
            y="trip_count",
            labels={"date": "Date", "trip_count": "Trips"},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(
            hour_pattern,
            x="hour",
            y="trip_count",
            color="day_type",
            markers=True,
            labels={"hour": "Hour", "trip_count": "Avg trips", "day_type": ""},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


def show_zone_analysis(features: pd.DataFrame) -> None:
    top_zones = (
        features.groupby(["zone", "borough"], as_index=False)["trip_count"]
        .sum()
        .sort_values("trip_count", ascending=False)
        .head(15)
    )
    zone_hour = (
        features.groupby(["zone", "hour"], as_index=False)["trip_count"]
        .mean()
        .sort_values(["zone", "hour"])
    )

    left, right = st.columns([1, 1])
    with left:
        fig = px.bar(
            top_zones,
            x="trip_count",
            y="zone",
            color="borough",
            orientation="h",
            labels={"trip_count": "Trips", "zone": "Zone", "borough": "Borough"},
        )
        fig.update_layout(
            height=520,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        selected = top_zones["zone"].head(5).tolist()
        fig = px.line(
            zone_hour[zone_hour["zone"].isin(selected)],
            x="hour",
            y="trip_count",
            color="zone",
            labels={"hour": "Hour", "trip_count": "Avg trips", "zone": "Zone"},
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


def show_weather_impact(features: pd.DataFrame) -> None:
    rain_summary = (
        features.groupby("is_raining", as_index=False)["trip_count"]
        .mean()
        .replace({"is_raining": {False: "No rain", True: "Rain"}})
    )
    sampled = features.sample(min(len(features), 10_000), random_state=42)

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            rain_summary,
            x="is_raining",
            y="trip_count",
            labels={"is_raining": "", "trip_count": "Avg trips per zone-hour"},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(
            sampled,
            x="temperature",
            y="trip_count",
            color="is_raining",
            opacity=0.35,
            labels={
                "temperature": "Temperature",
                "trip_count": "Trips",
                "is_raining": "Raining",
            },
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


def show_model_performance() -> None:
    metrics = load_model_metrics()
    if metrics.empty:
        st.info("Run `python3 -m src.models.train` to create model metrics.")
        return

    metrics = metrics.sort_values("mae")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    fig = px.bar(
        metrics,
        x="model_name",
        y="mae",
        labels={"model_name": "Model", "mae": "MAE"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    predictions = load_model_predictions()
    if predictions.empty:
        st.info("No Postgres model predictions found yet.")
        return

    actual = predictions[predictions["actual_trip_count"].notna()]
    future = predictions[predictions["actual_trip_count"].isna()]

    if not actual.empty:
        actual_hourly = (
            actual.groupby("datetime_hour", as_index=False)
            .agg(
                actual_trip_count=("actual_trip_count", "sum"),
                predicted_trip_count=("predicted_trip_count", "sum"),
            )
            .melt(
                id_vars="datetime_hour",
                value_vars=["actual_trip_count", "predicted_trip_count"],
                var_name="series",
                value_name="trip_count",
            )
        )
        actual_hourly["series"] = actual_hourly["series"].replace(
            {
                "actual_trip_count": "Actual",
                "predicted_trip_count": "Predicted",
            }
        )
        fig = px.line(
            actual_hourly,
            x="datetime_hour",
            y="trip_count",
            color="series",
            labels={"datetime_hour": "Time", "trip_count": "Trips", "series": ""},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if not future.empty:
        future_hourly = future.groupby("datetime_hour", as_index=False)[
            "predicted_trip_count"
        ].sum()
        fig = px.line(
            future_hourly,
            x="datetime_hour",
            y="predicted_trip_count",
            labels={
                "datetime_hour": "Future time",
                "predicted_trip_count": "Predicted trips",
            },
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


def build_prediction_row(
    features: pd.DataFrame,
    zone_name: str,
    prediction_time: datetime,
    weather_inputs: dict[str, float | bool],
) -> pd.DataFrame:
    zone_history = features[features["zone"] == zone_name].sort_values("datetime_hour")
    latest = zone_history.iloc[-1]
    holiday_match = features[
        features["datetime_hour"].dt.date == prediction_time.date()
    ]["is_holiday"]
    is_holiday = bool(holiday_match.max()) if not holiday_match.empty else False

    row = {
        "zone_id": int(latest["zone_id"]),
        "hour": prediction_time.hour,
        "day_of_week": prediction_time.weekday(),
        "month": prediction_time.month,
        "is_weekend": prediction_time.weekday() in {5, 6},
        "is_holiday": is_holiday,
        "is_peak_hour": prediction_time.hour in {7, 8, 9, 17, 18, 19},
        "lag_1_hour_demand": float(latest["trip_count"]),
        "lag_24_hour_demand": float(latest["lag_24_hour_demand"]),
        "rolling_3_hour_avg": float(zone_history["trip_count"].tail(3).mean()),
        "rolling_24_hour_avg": float(zone_history["trip_count"].tail(24).mean()),
        **weather_inputs,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def show_prediction_simulator(features: pd.DataFrame) -> None:
    model_path = latest_model_path()
    if model_path is None:
        st.info("Train a model before using the simulator.")
        return

    zones = sorted(features["zone"].dropna().unique())
    left, right = st.columns([1, 1])
    with left:
        default_zone_index = (
            zones.index("Midtown Center") if "Midtown Center" in zones else 0
        )
        zone_name = st.selectbox("Zone", zones, index=default_zone_index)
        selected_date = st.date_input(
            "Date", value=features["datetime_hour"].max().date()
        )
        selected_time = st.time_input(
            "Time",
            value=datetime.now().time().replace(minute=0, second=0, microsecond=0),
        )
        temperature = st.number_input(
            "Temperature", value=float(features["temperature"].median())
        )
        humidity = st.number_input(
            "Humidity",
            min_value=0.0,
            max_value=100.0,
            value=float(features["humidity"].median()),
        )
        precipitation = st.number_input("Precipitation", min_value=0.0, value=0.0)
        rain = st.number_input("Rain", min_value=0.0, value=0.0)
        wind_speed = st.number_input(
            "Wind speed", min_value=0.0, value=float(features["wind_speed"].median())
        )
    with right:
        prediction_time = datetime.combine(selected_date, selected_time)
        weather_inputs = {
            "temperature": temperature,
            "humidity": humidity,
            "precipitation": precipitation,
            "rain": rain,
            "wind_speed": wind_speed,
            "is_raining": precipitation > 0 or rain > 0,
        }
        row = build_prediction_row(features, zone_name, prediction_time, weather_inputs)
        model = load_trained_model(str(model_path))
        prediction = float(np.maximum(model.predict(row), 0.0)[0])
        st.metric("Predicted pickups", f"{prediction:,.1f}")
        st.write(f"Model: `{Path(model_path).name}`")
        st.dataframe(row, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Urban Mobility Forecasting")

    try:
        features, data_source = load_features()
    except FileNotFoundError:
        st.warning(
            "Load Postgres or run the pipeline to create "
            "`data/processed/feature_table.parquet`."
        )
        return

    min_date = features["datetime_hour"].min().date()
    max_date = features["datetime_hour"].max().date()

    with st.sidebar:
        st.header("Filters")
        st.caption(f"Data source: {data_source}")
        date_range = st.date_input("Date range", value=(min_date, max_date))
        if len(date_range) != 2:
            st.stop()

        borough_options = sorted(features["borough"].dropna().unique())
        boroughs = st.multiselect("Borough", borough_options, default=borough_options)
        zone_options = sorted(
            features[features["borough"].isin(boroughs)]["zone"].dropna().unique()
        )
        zones = st.multiselect("Zone", zone_options)

    filtered = filter_features(features, date_range, boroughs, zones)
    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    overview, trends, zones_tab, weather, models, simulator = st.tabs(
        [
            "Overview",
            "Demand Trends",
            "Zone Analysis",
            "Weather Impact",
            "Model Performance",
            "Prediction Simulator",
        ]
    )

    with overview:
        show_overview(filtered)
    with trends:
        show_trends(filtered)
    with zones_tab:
        show_zone_analysis(filtered)
    with weather:
        show_weather_impact(filtered)
    with models:
        show_model_performance()
    with simulator:
        show_prediction_simulator(features)


if __name__ == "__main__":
    main()
