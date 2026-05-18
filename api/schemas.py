from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ZoneResponse(BaseModel):
    location_id: int
    borough: str | None = None
    zone: str | None = None
    service_zone: str | None = None


class DemandResponse(BaseModel):
    datetime_hour: datetime
    zone_id: int
    trip_count: int


class PredictionRequest(BaseModel):
    zone_id: int
    datetime_hour: datetime
    temperature: float | None = None
    humidity: float | None = None
    precipitation: float | None = None
    rain: float | None = None
    wind_speed: float | None = None
    is_holiday: bool = False


class PredictionResponse(BaseModel):
    zone_id: int
    zone_name: str | None = None
    predicted_trip_count: float = Field(ge=0)
    model_name: str
    model_version: str
