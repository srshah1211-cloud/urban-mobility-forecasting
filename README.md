# Urban Mobility Forecasting

Predict hourly NYC taxi pickup demand by taxi zone.

## Current Data Window

This repo is configured for the local files currently present:

- `data/raw/yellow_tripdata_2026-01.parquet`
- `data/raw/yellow_tripdata_2026-02.parquet`
- `data/raw/yellow_tripdata_2026-03.parquet`
- `data/external/taxi_zone_lookup.csv`

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## External Data

Generate holidays:

```bash
python3 -m src.ingestion.load_holidays \
  --years 2026 \
  --output data/external/us_holidays_2026.csv
```

Fetch weather from Open-Meteo:

```bash
python3 -m src.ingestion.load_weather_data \
  --start-date 2026-01-01 \
  --end-date 2026-03-31 \
  --output data/external/weather_nyc_2026_01_03.csv
```

The weather fetch uses NYC coordinates, the hourly variables from the project plan,
and `timezone=America/New_York` so weather timestamps align with TLC taxi pickup
timestamps.

## Build Processed Data

After the weather CSV exists:

```bash
python3 -m src.pipeline \
  --start-date 2026-01-01 \
  --end-date 2026-03-31 \
  --weather-path data/external/weather_nyc_2026_01_03.csv \
  --holidays-path data/external/us_holidays_2026.csv
```

Outputs:

- `data/processed/hourly_zone_demand.parquet`
- `data/processed/feature_table.parquet`

## Train Models

```bash
python3 -m src.models.train
```

This trains a historical-average baseline, random forest, and XGBoost if
installed. By default it also writes model metrics, holdout predictions, and 30
days of future predictions into PostgreSQL when `DATABASE_URL` is reachable and
the base tables have already been loaded.

Useful options:

```bash
# Train and store 30 days of future predictions in Postgres
python3 -m src.models.train --future-days 30

# Train locally without writing model runs/predictions to Postgres
python3 -m src.models.train --no-store-to-postgres
```

## API

```bash
uvicorn api.main:app --reload
```

Endpoints:

- `GET /health`
- `GET /zones`
- `GET /demand/actual?zone_id=161&start_date=2026-01-01&end_date=2026-01-07`
- `POST /predictions`

The API reads from PostgreSQL when `DATABASE_URL` is reachable. If the database
is not running or the tables are empty, it falls back to local parquet/CSV files.

## PostgreSQL

Start Postgres:

```bash
docker compose up -d postgres
```

Load processed data into Postgres:

```bash
python3 -m src.database.load_to_postgres \
  --weather-path data/external/weather_nyc_2026_01_03.csv \
  --holidays-path data/external/us_holidays_2026.csv
```

Run the API and dashboard against Postgres:

```bash
docker compose up api dashboard
```

The dashboard also prefers PostgreSQL and falls back to
`data/processed/feature_table.parquet` for local development.

## One Command Data Load

After Postgres is running, this command generates holidays, uses the existing
weather CSV, rebuilds processed parquet files, applies the SQL schema, and loads
Postgres:

```bash
python3 -m src.run_all --use-existing-weather
```

To also train models and store dashboard-ready predictions:

```bash
python3 -m src.run_all --use-existing-weather --train --future-days 30
```

Without `--use-existing-weather`, the command fetches weather from Open-Meteo.

## Documentation

Project architecture and modeling details are documented in:

- `docs/architecture.md`
- `docs/modeling_approach.md`

## Tests

```bash
python3 -m pytest
```

## Screenshots
<img width="1510" height="810" alt="image" src="https://github.com/user-attachments/assets/97632059-e9e0-4ba8-b13c-e077b63a7c8c" />
<img width="1505" height="828" alt="image" src="https://github.com/user-attachments/assets/be83c842-8efb-45ab-a888-70ffd9c3da2c" />
<img width="1509" height="847" alt="image" src="https://github.com/user-attachments/assets/a20ffc2a-ffb5-4f9b-86cc-a329197786af" />
<img width="1508" height="855" alt="image" src="https://github.com/user-attachments/assets/2619ff2f-712e-4513-8812-c4e6530c1905" />
<img width="1487" height="846" alt="image" src="https://github.com/user-attachments/assets/fc7231bb-65cc-4ef4-aa66-08abb600b6f9" />
<img width="1497" height="838" alt="image" src="https://github.com/user-attachments/assets/9b601729-c88e-4881-a615-b88c223abf86" />






