from __future__ import annotations

from fastapi import FastAPI

from api.routes import demand, health, predictions, zones

app = FastAPI(title="Urban Mobility Forecasting API", version="0.1.0")

app.include_router(health.router)
app.include_router(zones.router)
app.include_router(demand.router)
app.include_router(predictions.router)
