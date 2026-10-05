"""
app.py
Flask REST API for house price prediction.

Endpoints:
  GET  /api/health          → liveness check
  POST /api/predict         → predict price from area + rooms
  GET  /api/metrics         → model evaluation metrics
  GET  /api/data            → raw dataset (for frontend charts)
  GET  /                    → serves frontend/index.html
"""
import os
import sys
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "house_price_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "..", "model", "metrics.json")

sys.path.insert(0, BASE_DIR)
from data_loader import load_data

app = Flask(__name__, static_folder=os.path.join(FRONTEND_DIR, "static"))
CORS(app)

# ── Load model at startup ──────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run 'python backend/train.py' first."
    )
model = joblib.load(MODEL_PATH)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, "static"), filename)


@app.route("/api/health")
def health():
    model_name = "Ridge Regression"
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                m = json.load(f)
                model_name = m.get("model_name", model_name)
        except Exception:
            pass
    return jsonify({"status": "ok", "model": model_name})


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    try:
        area = float(body["area"])
        rooms = int(body["rooms"])
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    if area <= 0 or rooms <= 0:
        return jsonify({"error": "area and rooms must be positive numbers"}), 400

    X = np.array([[area, rooms]])
    predicted_price = float(model.predict(X)[0])

    # confidence interval: ±1 RMSE (simple approximation)
    rmse = _get_rmse()
    low = max(0, predicted_price - rmse)
    high = predicted_price + rmse

    return jsonify({
        "area": area,
        "rooms": rooms,
        "predicted_price": round(predicted_price, 2),
        "price_range": {
            "low": round(low, 2),
            "high": round(high, 2),
        },
        "formatted": f"${predicted_price:,.0f}",
    })


@app.route("/api/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Metrics not found. Run train.py first."}), 404
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/data")
def data():
    df = load_data()
    return jsonify({
        "area": df["area"].tolist(),
        "rooms": df["rooms"].tolist(),
        "price": df["price"].tolist(),
    })


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_rmse() -> float:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        return m.get("rmse", 50000)
    return 50000


if __name__ == "__main__":
    app.run(debug=True, port=5000)
