from __future__ import annotations

from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.inference import predict_future

app = FastAPI(title="Sekka Congestion Forecast API", version="1.0")


class PredictRequest(BaseModel):
    route_id: str = Field(..., description="Route identifier, e.g., '123'")
    future_hours: int = Field(24, ge=1, le=24 * 14, description="Forecast horizon in hours (1-336)")


class ForecastPoint(BaseModel):
    timestamp: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class PredictResponse(BaseModel):
    route_id: str
    points: List[ForecastPoint]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        df = predict_future(route_id=req.route_id, future_hours=req.future_hours)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found for route_id {req.route_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    points = [
        ForecastPoint(
            timestamp=pd.Timestamp(row.ds).isoformat(),
            yhat=float(row.yhat),
            yhat_lower=float(row.yhat_lower),
            yhat_upper=float(row.yhat_upper),
        )
        for row in df.itertuples(index=False)
    ]

    return PredictResponse(route_id=req.route_id, points=points)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=config.API_HOST, port=config.API_PORT, reload=True)
