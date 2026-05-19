# Urban Mobility Forecasting Architecture

## Purpose

This project predicts hourly NYC yellow taxi pickup demand by taxi zone. It turns
raw taxi trips, taxi zone metadata, weather, and holiday data into a feature table,
trains forecasting models, stores results in PostgreSQL, exposes selected data
through FastAPI, and visualizes trends and predictions in Streamlit.

## High-Level Flow

```text
Raw TLC taxi parquet files
        +
Taxi zone lookup CSV
        +
Open-Meteo hourly weather
        +
US holiday calendar
        |
        v
Ingestion modules
        |
        v
Cleaning and hourly zone aggregation
        |
        v
Feature engineering
        |
        v
Processed parquet files
        |
        v
PostgreSQL tables
        |
        +--> Model training and prediction storage
        |
        +--> FastAPI endpoints
        |
        +--> Streamlit dashboard
```

## Main Components

### Data Sources

The project currently targets the January-March 2026 MVP window.

- `data/raw/yellow_tripdata_2026-01.parquet`
- `data/raw/yellow_tripdata_2026-02.parquet`
- `data/raw/yellow_tripdata_2026-03.parquet`
- `data/external/taxi_zone_lookup.csv`
- `data/external/weather_nyc_2026_01_03.csv`
- `data/external/us_holidays_2026.csv`

Taxi data provides the actual pickup demand. Weather and calendar data provide
external context that can influence demand.

### Ingestion Layer

Located in `src/ingestion/`.

- `load_taxi_data.py`: lists and reads TLC yellow taxi parquet files.
- `load_zone_lookup.py`: loads taxi zone metadata.
- `load_weather_data.py`: fetches hourly Open-Meteo archive data and writes CSV.
- `load_holidays.py`: generates US holidays using the `holidays` package when
  installed, with a Pandas federal-holiday fallback.

Weather is requested with `timezone=America/New_York` so hourly weather aligns
with taxi pickup timestamps.

### Processing Layer

Located in `src/processing/`.

- `clean_taxi_data.py`: filters invalid trips and creates `datetime_hour`.
- `aggregate_demand.py`: groups pickups by `datetime_hour` and `zone_id`.
- `join_external_data.py`: joins demand with zones, weather, and holidays.

The demand table is zero-filled across all taxi zones and all hours in the date
range. This matters because "zero trips" is a real training signal, not missing
data.

### Feature Layer

Located in `src/features/build_features.py`.

Calendar features:

- `hour`
- `day_of_week`
- `day_name`
- `month`
- `is_weekend`
- `is_peak_hour`

Demand history features:

- `lag_1_hour_demand`
- `lag_24_hour_demand`
- `rolling_3_hour_avg`
- `rolling_24_hour_avg`

These features let the model learn daily patterns, rush-hour patterns, weekend
effects, weather effects, and recent-demand momentum.

### Processed Files

The local pipeline writes:

- `data/processed/hourly_zone_demand.parquet`
- `data/processed/feature_table.parquet`

These files are useful for local development, reproducibility, and fallback
behavior when PostgreSQL is not available.

### PostgreSQL Layer

SQL source files live in `sql/`.

- `001_create_tables.sql`: creates database tables and constraints.
- `002_create_indexes.sql`: creates indexes for faster lookup.
- `003_seed_static_tables.sql`: placeholder for future static seed data.

Database helpers live in `src/database/`.

- `schema.py`: applies SQL schema files.
- `connection.py`: creates a SQLAlchemy engine using `DATABASE_URL`.
- `load_to_postgres.py`: loads static and processed data.
- `read.py`: reads dashboard/API data from PostgreSQL.
- `write.py`: stores model runs and predictions.

Primary tables:

- `taxi_zones`
- `weather_hourly`
- `holidays`
- `hourly_zone_demand`
- `feature_table`
- `model_runs`
- `model_predictions`

The application prefers PostgreSQL when reachable, but keeps parquet fallback for
local development.

### Training Layer

Located in `src/models/`.

- `config.py`: defines feature columns.
- `train.py`: trains models, evaluates metrics, saves best model, stores outputs.
- `evaluate.py`: computes MAE, RMSE, MAPE, and R2.
- `predict.py`: loads saved models and predicts from a feature frame.
- `registry.py`: finds the latest saved model file.

Training stores:

- local model file in `models/`
- `models/feature_columns.json`
- `models/model_metrics.csv`
- PostgreSQL rows in `model_runs`
- PostgreSQL rows in `model_predictions`

### API Layer

Located in `api/`.

FastAPI exposes:

- `GET /health`
- `GET /zones`
- `GET /demand/actual`
- `POST /predictions`

The current prediction API uses a historical average baseline. The dashboard
prediction simulator directly uses the saved model file.

### Dashboard Layer

Located in `dashboard/app.py`.

The Streamlit dashboard includes:

- Overview KPIs
- Demand trends
- Zone analysis
- Weather impact
- Model performance
- Prediction simulator

The dashboard reads from PostgreSQL first. If PostgreSQL is unavailable, it falls
back to local parquet files.

## One-Command Runtime

The orchestration entrypoint is `src/run_all.py`.

```bash
python3 -m src.run_all --use-existing-weather --train --future-days 30
```

This performs:

1. Generate holidays.
2. Use or fetch weather.
3. Build processed parquet files.
4. Apply PostgreSQL schema.
5. Load PostgreSQL tables.
6. Optionally train models.
7. Optionally store model runs and predictions.

PostgreSQL must already be running and reachable through `DATABASE_URL`.

## Deployment Shape

`docker-compose.yml` defines:

- `postgres`
- `api`
- `dashboard`

Configuration values come from `.env`.

Local ports:

- FastAPI: `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`
- PostgreSQL: `localhost:5432`

## Data Ownership

Raw input files are treated as source data. Processed parquet files and database
tables are generated artifacts. Model files and prediction rows are generated
from the feature table and can be recreated by rerunning the pipeline and
training command.

