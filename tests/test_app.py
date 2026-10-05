"""
test_app.py
Unit and integration tests for the House Price Prediction project.
"""
import os
import sys
import json
import joblib
import pytest
import numpy as np

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from data_loader import load_data
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_data_loader():
    """Verify data_loader reads house_prices.xlsx correctly."""
    df = load_data()
    assert len(df) > 0, "DataFrame should not be empty"
    assert set(["area", "rooms", "price"]).issubset(df.columns)
    assert (df["area"] > 0).all()
    assert (df["rooms"] > 0).all()
    assert (df["price"] > 0).all()


def test_model_artifact():
    """Verify saved model artifact makes accurate predictions."""
    model_path = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
    assert os.path.exists(model_path), "Model artifact file missing"

    model = joblib.load(model_path)
    prediction = model.predict(np.array([[1800, 3]]))
    assert len(prediction) == 1
    assert prediction[0] > 0, "Prediction price should be positive"


def test_metrics_json():
    """Verify metrics.json contains required model statistics."""
    metrics_path = os.path.join(BASE_DIR, "model", "metrics.json")
    assert os.path.exists(metrics_path), "metrics.json missing"

    with open(metrics_path) as f:
        metrics = json.load(f)

    for field in ["model_name", "mae", "rmse", "r2", "cv_r2_mean"]:
        assert field in metrics, f"Missing metric field: {field}"


def test_api_health(client):
    """Test /api/health endpoint."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "model" in data


def test_api_predict_valid(client):
    """Test /api/predict endpoint with valid data."""
    payload = {"area": 2000, "rooms": 4}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0
    assert "price_range" in data
    assert "low" in data["price_range"]
    assert "high" in data["price_range"]


def test_api_predict_invalid(client):
    """Test /api/predict endpoint with invalid input."""
    payload = {"area": -500, "rooms": 0}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_api_metrics(client):
    """Test /api/metrics endpoint."""
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.get_json()
    assert "r2" in data


def test_api_data(client):
    """Test /api/data endpoint."""
    res = client.get("/api/data")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["area"]) == len(data["price"])


def test_index_route(client):
    """Test serving index.html on root route."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"House Price Predictor" in res.data
