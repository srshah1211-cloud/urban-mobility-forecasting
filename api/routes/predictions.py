from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import get_feature_table, get_zones
from api.schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["predictions"])


@router.post("/predictions", response_model=PredictionResponse)
def predict_demand(request: PredictionRequest) -> PredictionResponse:
    features = get_feature_table()
    hour = request.datetime_hour.hour
    comparable = features[
        (features["zone_id"] == request.zone_id) & (features["hour"] == hour)
    ]
    if comparable.empty:
        prediction = 0.0
    else:
        prediction = float(comparable["trip_count"].mean())

    zones = get_zones()
    zone_match = zones[zones["location_id"] == request.zone_id]
    zone_name = None if zone_match.empty else str(zone_match.iloc[0]["zone"])
    return PredictionResponse(
        zone_id=request.zone_id,
        zone_name=zone_name,
        predicted_trip_count=max(prediction, 0.0),
        model_name="historical_hourly_average",
        model_version="v0",
    )
