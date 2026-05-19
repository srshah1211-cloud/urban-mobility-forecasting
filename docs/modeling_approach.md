# Modeling Approach and Prediction Logic

## Goal

The model predicts:

```text
trip_count
```

for each:

```text
taxi zone + hour
```

In plain terms, the model answers:

```text
How many yellow taxi pickups should we expect in this zone during this hour?
```

## Training Data

The model is trained from `data/processed/feature_table.parquet` or the
equivalent PostgreSQL `feature_table`.

Each row represents one zone-hour example:

```text
datetime_hour = 2026-01-01 08:00
zone_id = 161
weather = cold, dry, windy
calendar = Thursday, holiday, peak hour
recent demand = prior hour and rolling demand values
target = actual trip_count
```

The model sees many historical examples like this and learns which combinations
of time, zone, weather, and recent demand usually lead to higher or lower taxi
pickup counts.

## Feature Columns

The model input columns are defined in `src/models/config.py`.

```text
zone_id
hour
day_of_week
month
is_weekend
is_holiday
is_peak_hour
temperature
humidity
precipitation
rain
wind_speed
is_raining
lag_1_hour_demand
lag_24_hour_demand
rolling_3_hour_avg
rolling_24_hour_avg
```

These were chosen because taxi demand has strong patterns across:

- location
- hour of day
- weekday versus weekend
- holiday behavior
- weather conditions
- recent demand momentum

## Current Models

The training script evaluates:

- historical average baseline
- Random Forest Regressor
- XGBoost Regressor, if installed

The best model is selected by lowest MAE.

MAE means:

```text
On average, how many pickups is the model off by?
```

This is the most understandable metric for taxi demand forecasting.

## Why Random Forest

Random Forest is a strong choice for this MVP because it:

- works well on tabular data
- handles nonlinear relationships
- handles mixed feature types like zone, hour, weather, and lags
- needs less preprocessing than linear models
- is easier to train and debug than deep learning
- performs well with limited data

It is also robust for a first production-style model. The project currently has
three months of hourly data, which is enough for tree-based models but not ideal
for more complex sequence models.

Random Forest does not use one simple formula. It builds many decision trees.
Each tree asks learned questions such as:

```text
Is this a high-demand zone?
Is the hour during peak commute?
Was demand high in the previous hour?
Is it raining?
```

Each tree gives a prediction. The forest averages those predictions.

Conceptually:

```text
final prediction = average(prediction from tree 1, tree 2, ..., tree N)
```

## Why XGBoost

XGBoost is included because it is often very strong for tabular forecasting
problems.

It can capture:

- nonlinear effects
- feature interactions
- zone-specific demand patterns
- lag and rolling-demand behavior

Compared with Random Forest, XGBoost often gives better accuracy after tuning.
For the MVP, it is optional because it adds another dependency and can take more
tuning work.

## Why Historical Average Baseline

The historical average model is not meant to be the final model. It is a sanity
check.

It answers:

```text
For this zone and hour, what is the average historical demand?
```

Every more advanced model should beat this baseline. If it does not, the feature
engineering or training setup needs review.

## Why Not ARIMA

ARIMA is designed mainly for one time series at a time.

This project has many related time series:

```text
one hourly demand series per taxi zone
```

It also uses external features:

- weather
- holidays
- zone metadata
- rolling demand

ARIMA can be extended with external regressors, but managing it across hundreds
of zones becomes awkward. For this use case, tree-based tabular models are more
practical.

## Why Not LSTM First

LSTM models can be useful for sequence forecasting, but they are not the best
first choice here.

Reasons:

- only three months of current data are available
- LSTMs need careful sequence-window construction
- LSTMs need more tuning and validation
- LSTMs are easier to overfit on limited data
- the current tabular features already capture important time behavior

An LSTM can be added later after the Random Forest/XGBoost pipeline is stable and
more historical data is available.

## Prediction Simulator

The dashboard prediction simulator builds a single row of model inputs from:

- selected zone
- selected date and time
- user-entered weather
- latest historical demand for that zone
- recent rolling demand for that zone

Then it loads the saved model:

```text
models/random_forest_model.pkl
```

and calls:

```python
model.predict(row)
```

The output is clipped at zero because negative taxi pickups are impossible.

## Future Predictions

When training is run with:

```bash
python3 -m src.models.train --future-days 30
```

the training script can store future prediction rows in PostgreSQL.

Future prediction rows have:

```text
actual_trip_count = NULL
predicted_trip_count = model output
```

Because future actual demand is unknown, these rows are used for forecast
visualization.

The future prediction process uses:

- calendar features from the future timestamp
- holidays generated from the holiday calendar
- historical weather averages
- recursive demand lags from recent actual/predicted demand

This means future predictions are based on past patterns. They are not live
weather forecasts unless a future weather source is added later.

## Model Artifacts

Training writes:

```text
models/random_forest_model.pkl
models/feature_columns.json
models/model_metrics.csv
```

The `.pkl` file stores the trained model object. The JSON file stores the exact
feature order expected by the model. The CSV stores evaluation metrics.

When PostgreSQL is available, training also writes:

```text
model_runs
model_predictions
```

These tables allow the dashboard to show model performance and future prediction
trends.

## Main Limitations

Current limitations:

- only three months of taxi history
- future weather is estimated from historical averages
- the prediction API still uses a baseline rather than the saved model
- zone ID is treated as a numeric feature rather than a categorical embedding
- model tuning is minimal

Recommended next improvements:

- add more months of taxi data
- tune Random Forest and XGBoost
- add saved-model inference to the API
- add feature importance plots
- add backtesting by month
- add optional real weather forecast ingestion

