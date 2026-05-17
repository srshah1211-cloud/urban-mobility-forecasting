CREATE TABLE IF NOT EXISTS taxi_zones (
    location_id INTEGER PRIMARY KEY,
    borough VARCHAR(100),
    zone VARCHAR(150),
    service_zone VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS weather_hourly (
    id SERIAL PRIMARY KEY,
    datetime_hour TIMESTAMP NOT NULL UNIQUE,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    rain DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    is_raining BOOLEAN
);

CREATE TABLE IF NOT EXISTS holidays (
    date DATE PRIMARY KEY,
    holiday_name VARCHAR(150),
    is_holiday BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS hourly_zone_demand (
    id BIGSERIAL PRIMARY KEY,
    datetime_hour TIMESTAMP NOT NULL,
    zone_id INTEGER NOT NULL REFERENCES taxi_zones(location_id),
    trip_count INTEGER NOT NULL,
    avg_trip_distance DOUBLE PRECISION,
    avg_fare DOUBLE PRECISION,
    avg_total_amount DOUBLE PRECISION,
    avg_passenger_count DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_hourly_zone_demand UNIQUE (datetime_hour, zone_id),
    CONSTRAINT chk_trip_count_non_negative CHECK (trip_count >= 0)
);

CREATE TABLE IF NOT EXISTS feature_table (
    id BIGSERIAL PRIMARY KEY,
    datetime_hour TIMESTAMP NOT NULL,
    zone_id INTEGER NOT NULL REFERENCES taxi_zones(location_id),
    borough VARCHAR(100),
    zone_name VARCHAR(150),
    trip_count INTEGER NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    rain DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    is_raining BOOLEAN,
    hour INTEGER,
    day_of_week INTEGER,
    month INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    is_peak_hour BOOLEAN,
    lag_1_hour_demand DOUBLE PRECISION,
    lag_24_hour_demand DOUBLE PRECISION,
    rolling_3_hour_avg DOUBLE PRECISION,
    rolling_24_hour_avg DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_feature_table UNIQUE (datetime_hour, zone_id),
    CONSTRAINT chk_feature_trip_count_non_negative CHECK (trip_count >= 0),
    CONSTRAINT chk_hour_range CHECK (hour >= 0 AND hour <= 23),
    CONSTRAINT chk_day_of_week_range CHECK (day_of_week >= 0 AND day_of_week <= 6),
    CONSTRAINT chk_month_range CHECK (month >= 1 AND month <= 12)
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    train_start_date DATE,
    train_end_date DATE,
    test_start_date DATE,
    test_end_date DATE,
    mae DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    mape DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    model_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id BIGSERIAL PRIMARY KEY,
    model_run_id INTEGER REFERENCES model_runs(model_run_id),
    datetime_hour TIMESTAMP NOT NULL,
    zone_id INTEGER NOT NULL REFERENCES taxi_zones(location_id),
    actual_trip_count INTEGER,
    predicted_trip_count DOUBLE PRECISION,
    prediction_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_model_prediction UNIQUE (model_run_id, datetime_hour, zone_id)
);
