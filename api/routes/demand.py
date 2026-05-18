from __future__ import annotations

from datetime import date
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Query

from api.dependencies import get_hourly_demand
from api.schemas import DemandResponse

router = APIRouter(prefix="/demand", tags=["demand"])


@router.get("/actual", response_model=list[DemandResponse])
def actual_demand(
    zone_id: Annotated[int, Query(ge=1)],
    start_date: date,
    end_date: date,
) -> list[dict]:
    demand = get_hourly_demand()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    filtered = demand[
        (demand["zone_id"] == zone_id)
        & (demand["datetime_hour"] >= start)
        & (demand["datetime_hour"] < end)
    ]
    return filtered[["datetime_hour", "zone_id", "trip_count"]].to_dict(
        orient="records"
    )
