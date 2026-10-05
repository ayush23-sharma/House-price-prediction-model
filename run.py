"""
run.py  — launch the Flask development server from the project root.
Usage:  python run.py
"""
import sys
import os

# Make sure 'backend' is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting House Price Predictor at http://localhost:{port}")
    app.run(debug=False, port=port, host="0.0.0.0")
