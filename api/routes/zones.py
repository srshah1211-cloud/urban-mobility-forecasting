from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import get_zones
from api.schemas import ZoneResponse

router = APIRouter(tags=["zones"])


@router.get("/zones", response_model=list[ZoneResponse])
def list_zones() -> list[dict]:
    zones = get_zones().sort_values(["borough", "zone"])
    return zones.to_dict(orient="records")
