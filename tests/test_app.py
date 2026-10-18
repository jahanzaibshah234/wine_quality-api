
import importlib
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
    # Prevent the test from downloading a model from S3.
    monkeypatch.setattr(
        download_model,
        "download_model",
        lambda: None,
    )

    # Replace the real model with a predictable test model.
    monkeypatch.setattr(
        joblib,
        "load",
        lambda path: FakeModel(),
    )

    # Import the app only after replacing its dependencies.
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    with TestClient(app_module.app) as test_client:
        yield test_client

    sys.modules.pop("app", None)


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model": "wine_quality",
    }


def test_predict_endpoint(client):
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


def test_predict_rejects_missing_feature(client):
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
