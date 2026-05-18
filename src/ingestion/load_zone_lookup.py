from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import paths

ZONE_COLUMNS = {
    "LocationID": "location_id",
    "Borough": "borough",
    "Zone": "zone",
    "service_zone": "service_zone",
}


def load_zone_lookup(path: Path = paths.zone_lookup) -> pd.DataFrame:
    zones = pd.read_csv(path)
    missing = set(ZONE_COLUMNS) - set(zones.columns)
    if missing:
        raise ValueError(f"Taxi zone lookup is missing columns: {sorted(missing)}")
    zones = zones[list(ZONE_COLUMNS)].rename(columns=ZONE_COLUMNS)
    zones["location_id"] = zones["location_id"].astype("int64")
    return zones
