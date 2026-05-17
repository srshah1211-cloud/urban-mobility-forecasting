CREATE INDEX IF NOT EXISTS idx_weather_hourly_datetime
ON weather_hourly(datetime_hour);

CREATE INDEX IF NOT EXISTS idx_hourly_zone_demand_datetime
ON hourly_zone_demand(datetime_hour);

CREATE INDEX IF NOT EXISTS idx_hourly_zone_demand_zone
ON hourly_zone_demand(zone_id);

CREATE INDEX IF NOT EXISTS idx_hourly_zone_demand_zone_datetime
ON hourly_zone_demand(zone_id, datetime_hour);

CREATE INDEX IF NOT EXISTS idx_feature_table_datetime
ON feature_table(datetime_hour);

CREATE INDEX IF NOT EXISTS idx_feature_table_zone
ON feature_table(zone_id);

CREATE INDEX IF NOT EXISTS idx_feature_table_zone_datetime
ON feature_table(zone_id, datetime_hour);

CREATE INDEX IF NOT EXISTS idx_feature_table_borough
ON feature_table(borough);

CREATE INDEX IF NOT EXISTS idx_model_predictions_datetime
ON model_predictions(datetime_hour);

CREATE INDEX IF NOT EXISTS idx_model_predictions_zone
ON model_predictions(zone_id);

CREATE INDEX IF NOT EXISTS idx_model_predictions_model_run
ON model_predictions(model_run_id);
