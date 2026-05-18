from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import settings


def get_engine(database_url: str = settings.database_url) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)
