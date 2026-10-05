# House Price Prediction

A full-stack house price prediction app powered by Python, scikit-learn, and Flask.

## Dataset

`data/house_prices.xlsx` — 47 samples with 3 features:

| Column | Description |
|--------|-------------|
| `area` | House area in square feet |
| `rooms` | Number of rooms |
| `price` | Sale price in USD |

## Project Structure

```
house price prediction/
├── data/
│   └── house_prices.xlsx        # Source dataset
├── backend/
│   ├── data_loader.py           # Reads & cleans xlsx data
│   ├── train.py                 # Trains model, saves artifacts, generates plots
│   └── app.py                   # Flask REST API
├── frontend/
│   ├── index.html               # Single-page UI
│   └── static/                  # Generated EDA/performance plots (PNG)
├── model/
│   ├── house_price_model.pkl    # Trained sklearn Pipeline
│   └── metrics.json             # MAE, RMSE, R² evaluation results
├── requirements.txt
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python backend/train.py
```

This will:
- Load `data/house_prices.xlsx`
- Train a **Linear Regression** model (with StandardScaler pipeline)
- Evaluate on an 80/20 train/test split + 5-fold cross-validation
- Save `model/house_price_model.pkl` and `model/metrics.json`
- Generate EDA charts in `frontend/static/`

### 4. Start the Flask server

```bash
python backend/app.py
```

Open your browser at **http://localhost:5000**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the frontend |
| GET | `/api/health` | Liveness check |
| POST | `/api/predict` | Predict house price |
| GET | `/api/metrics` | Model evaluation metrics |
| GET | `/api/data` | Raw dataset (JSON) |

### POST /api/predict

**Request**
```json
{ "area": 1800, "rooms": 3 }
```

**Response**
```json
{
  "area": 1800,
  "rooms": 3,
  "predicted_price": 347521.50,
  "price_range": { "low": 289000.00, "high": 406000.00 },
  "formatted": "$347,522"
}
```

## ML Pipeline

```
StandardScaler → LinearRegression
```

- **Features**: `area` (sq ft), `rooms`
- **Target**: `price` (USD)
- **Evaluation**: MAE, RMSE, R², 5-fold CV R²
- **Confidence interval**: ±1 RMSE around the prediction

## Technology Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, NumPy, pandas |
| Data | openpyxl (xlsx reading) |
| Backend | Flask, flask-cors |
| Visualisation | matplotlib, seaborn |
| Frontend | Vanilla HTML/CSS/JS, Chart.js |
| Model persistence | joblib |
