
import json
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from download_model import download_model
from fastapi import FastAPI, Request
from pydantic import BaseModel


# Configure one-line JSON logs for Docker stdout.
logger = logging.getLogger("wine_quality_api")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


METRIC_NAMESPACE = "WineQualityAPI"
SERVICE_NAME = "wine-quality-api"


def log_event(event: str, **fields):
    """Write a structured JSON event to the application log."""
    log_data = {
        "event": event,
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        **fields,
    }

    logger.info(json.dumps(log_data, allow_nan=False))


def emit_metrics(metrics: dict[str, float]):
    """Emit CloudWatch Embedded Metric Format (EMF) JSON."""
    units = {
        "APIRequests": "Count",
        "API5xxErrors": "Count",
        "PredictionLatencyMs": "Milliseconds",
        "PredictionFailures": "Count",
    }

    metric_definitions = [
        {"Name": name, "Unit": units[name]}
        for name in metrics
    ]

    emf_event = {
        "_aws": {
            "Timestamp": time.time_ns() // 1_000_000,
            "CloudWatchMetrics": [
                {
                    "Namespace": METRIC_NAMESPACE,
                    "Dimensions": [["Service"]],
                    "Metrics": metric_definitions,
                }
            ],
        },
        "Service": SERVICE_NAME,
        **metrics,
    }

    logger.info(json.dumps(emf_event, allow_nan=False))


app = FastAPI(title="Wine Quality Prediction API")

MODEL_NAME = "wine_quality"
MODEL_PATH = "wine_quality.pkl"

# Keep the existing model download and loading behavior.
download_model()
model = joblib.load(MODEL_PATH)


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Track request IDs, status codes, latency, and request metrics."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception as exc:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        log_event(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
        )

        emit_metrics({
            "APIRequests": 1,
            "API5xxErrors": 1,
        })

        raise

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    response.headers["X-Request-ID"] = request_id

    log_event(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )

    request_metrics = {"APIRequests": 1}

    if response.status_code >= 500:
        request_metrics["API5xxErrors"] = 1

    emit_metrics(request_metrics)

    return response


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
    }


@app.post("/predict")
def predict(features: WineFeatures, request: Request):
    start_time = time.perf_counter()
    request_id = request.state.request_id

    data = pd.DataFrame(
        [
            {
                "fixed acidity": features.fixed_acidity,
                "volatile acidity": features.volatile_acidity,
                "citric acid": features.citric_acid,
                "residual sugar": features.residual_sugar,
                "chlorides": features.chlorides,
                "free sulfur dioxide": features.free_sulfur_dioxide,
                "total sulfur dioxide": features.total_sulfur_dioxide,
                "density": features.density,
                "pH": features.pH,
                "sulphates": features.sulphates,
                "alcohol": features.alcohol,
            }
        ]
    )

    try:
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        confidence = float(max(probabilities))

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        log_event(
            "prediction_completed",
            request_id=request_id,
            model=MODEL_NAME,
            prediction=str(prediction),
            confidence=confidence,
            status_code=200,
            duration_ms=duration_ms,
        )

        emit_metrics({
            "PredictionLatencyMs": duration_ms,
        })

        return {
            "prediction": prediction,
            "confidence": confidence,
        }

    except Exception as exc:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        log_event(
            "prediction_failed",
            request_id=request_id,
            model=MODEL_NAME,
            status_code=500,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
        )

        emit_metrics({
            "PredictionLatencyMs": duration_ms,
            "PredictionFailures": 1,
        })

        raise

