from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import paths


def load_model(model_path: Path):
    return joblib.load(model_path)


def load_feature_columns(
    path: Path = paths.models / "feature_columns.json",
) -> list[str]:
    with path.open() as file:
        return json.load(file)


def predict_frame(
    frame: pd.DataFrame, model_path: Path, columns_path: Path
) -> pd.Series:
    model = load_model(model_path)
    columns = load_feature_columns(columns_path)
    x = frame[columns].copy()
    bool_columns = x.select_dtypes(include=["bool"]).columns
    x[bool_columns] = x[bool_columns].astype("int8")
    x = x.fillna(0)
    predictions = np.maximum(model.predict(x), 0.0)
    return pd.Series(predictions, index=frame.index, name="predicted_trip_count")
