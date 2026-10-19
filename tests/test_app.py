
import importlib
import json
import logging
import sys

import download_model
import joblib
import pytest
from fastapi.testclient import TestClient


class FakeModel:
    """A lightweight model for API tests."""

    def predict(self, data):
        return ["Good"]

    def predict_proba(self, data):
        return [[0.1, 0.2, 0.7]]


@pytest.fixture
def client(monkeypatch):
    # Prevent tests from downloading the model from S3.
    monkeypatch.setattr(
        download_model,
        "download_model",
        lambda: None,
    )

    # Use a predictable fake model.
    monkeypatch.setattr(
        joblib,
        "load",
        lambda path: FakeModel(),
    )

    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    with TestClient(app_module.app) as test_client:
        yield test_client

    sys.modules.pop("app", None)


def get_json_events(caplog):
    """Extract structured JSON events from application logs."""
    events = []

    for record in caplog.records:
        if record.name == "wine_quality_api":
            events.append(json.loads(record.getMessage()))

    return events

def get_emf_events(caplog):
    """Extract CloudWatch Embedded Metric Format events."""
    return [
        event
        for event in get_json_events(caplog)
        if "_aws" in event
    ]

def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model": "wine_quality",
    }

    assert response.headers.get("X-Request-ID")


def test_predict_endpoint(client, caplog):
    caplog.set_level(logging.INFO, logger="wine_quality_api")

    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == "Good"
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence"] == pytest.approx(0.7)

    request_id = response.headers["X-Request-ID"]
    events = get_json_events(caplog)

    prediction_events = [
        event
        for event in events
        if event.get("event") == "prediction_completed"
    ]

    assert len(prediction_events) == 1

    prediction_event = prediction_events[0]

    assert prediction_event["request_id"] == request_id
    assert prediction_event["model"] == "wine_quality"
    assert prediction_event["prediction"] == "Good"
    assert prediction_event["status_code"] == 200
    assert prediction_event["confidence"] == pytest.approx(0.7)
    assert prediction_event["duration_ms"] >= 0

    request_events = [
        event
        for event in events
        if event.get("event") == "request_completed"
    ]

    assert any(
        event["request_id"] == request_id
        and event["path"] == "/predict"
        and event["status_code"] == 200
        for event in request_events
    )

    emf_events = get_emf_events(caplog)

    latency_events = [
        event
        for event in emf_events
        if "PredictionLatencyMs" in event
    ]

    assert len(latency_events) == 1

    latency_event = latency_events[0]

    assert latency_event["Service"] == "wine-quality-api"
    assert latency_event["PredictionLatencyMs"] >= 0

    cloudwatch_config = latency_event["_aws"]["CloudWatchMetrics"][0]

    assert cloudwatch_config["Namespace"] == "WineQualityAPI"
    assert cloudwatch_config["Dimensions"] == [["Service"]]
    assert "request_id" not in cloudwatch_config["Dimensions"][0]

    request_metric_events = [
        event
        for event in emf_events
        if "APIRequests" in event
    ]

    assert any(
        event["APIRequests"] == 1
        and event["Service"] == "wine-quality-api"
        for event in request_metric_events
    )

    # Ensure raw feature values aren't included in structured events.
    for event in events:
        assert "fixed_acidity" not in event
        assert "volatile_acidity" not in event


def test_predict_rejects_missing_feature(client, caplog):
    caplog.set_level(logging.INFO, logger="wine_quality_api")

    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

    events = get_json_events(caplog)

    assert any(
        event.get("event") == "request_completed"
        and event["path"] == "/predict"
        and event["status_code"] == 422
        for event in events
    )

    # Validation fails before the prediction function executes.
    assert not any(
        event.get("event") == "prediction_completed"
        for event in events
    )

    emf_events = get_emf_events(caplog)

    assert any(
        event.get("APIRequests") == 1
        for event in emf_events
    )

    # HTTP 422 is a client-side validation error, not a server error.
    assert not any(
        event.get("API5xxErrors") == 1
        for event in emf_events
    )
