from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_detection.config import MODEL_FILENAME, MODELS_DIR
from fraud_detection.modeling import load_model


class TransactionRequest(BaseModel):
    amount: float = Field(ge=0)
    time: float = Field(ge=0)
    features: dict[str, float] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold: float
    model_path: str


def create_app(model_path: str | Path = MODELS_DIR / MODEL_FILENAME) -> FastAPI:
    app = FastAPI(
        title="Fraud Detection Prediction API",
        version="1.0.0",
        description="Small FastAPI wrapper around the trained fraud detection model.",
    )
    path = Path(model_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: TransactionRequest) -> PredictionResponse:
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Run training before starting the API.",
            )

        artifact: dict[str, Any] = load_model(path)
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        row = {feature: 0.0 for feature in feature_columns}
        row.update(payload.features)
        if "Amount" in row:
            row["Amount"] = payload.amount
        if "Time" in row:
            row["Time"] = payload.time

        frame = pd.DataFrame([row], columns=feature_columns)
        probability = float(model.predict_proba(frame)[0, 1])
        threshold = 0.5
        return PredictionResponse(
            is_fraud=probability >= threshold,
            fraud_probability=probability,
            threshold=threshold,
            model_path=str(path),
        )

    return app


app = create_app()
