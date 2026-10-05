# House Price Prediction — Project Completion Report

The full-stack House Price Prediction project has been completed and validated end-to-end.

---

## 1. Project Architecture & Structure

```
house price prediction/
├── house_prices.xlsx           # Source dataset (47 rows: area, rooms, price)
├── data/
│   └── house_prices.xlsx       # Project copy of dataset
├── backend/
│   ├── data_loader.py          # Data ingestion & cleaning utility
│   ├── train.py                # Multi-model ML training, CV evaluation & EDA generator
│   └── app.py                  # Flask REST API endpoints
├── frontend/
│   ├── index.html              # Modern dark glassmorphic single-page dashboard
│   └── static/                 # Generated plot images (PNG)
│       ├── area_vs_price.png
│       ├── rooms_vs_price.png
│       ├── correlation.png
│       ├── model_performance.png
│       └── model_comparison.png
├── model/
│   ├── house_price_model.pkl   # Serialized scikit-learn pipeline (Ridge Regression)
│   └── metrics.json            # Model evaluation & candidate comparison metrics
├── tests/
│   └── test_app.py             # Pytest unit and integration test suite (9 tests)
├── .venv/                      # Python 3.12 Virtual Environment
├── run.py                      # Application launcher
├── requirements.txt            # Dependency list
└── README.md                   # Setup and usage documentation
```

---

## 2. Machine Learning Pipeline & Model Selection

Following machine learning best practices (`ml-best-practices`), four regression algorithms were trained and benchmarked using 5-fold cross-validation (`CV R²`) on an 80/20 train/test split:

| Model Candidate | 5-Fold CV R² (Mean ± Std) | Test R² Score | MAE ($) | Evaluation Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **Ridge Regression** | **39.67% ± 45.83%** | **0.5183** | **$70,814** | **Selected (Best Generalization)** |
| **Linear Regression** | 36.39% ± 50.73% | 0.5149 | $72,335 | Candidate |
| **Random Forest** | -22.88% ± 131.70% | 0.5219 | $72,252 | Overfitting on 47 samples |
| **Gradient Boosting** | -76.78% ± 182.69% | 0.4957 | $72,593 | Overfitting on 47 samples |

> [!TIP]
> **Model Selection Rationale**: On this 47-sample dataset, **Ridge Regression** achieved the highest 5-fold cross-validation R² score ($39.67\%$) and the lowest Mean Absolute Error ($\$70,814$), protecting against multicollinearity between area and room count.

---

## 3. Frontend & User Interface

The frontend (`frontend/index.html`) is built as a single-page dashboard:
- **Design System**: Dark glassmorphism palette (`#0f172a`, `#1e1b4b`), Plus Jakarta Sans typography, sleek card borders, and smooth transitions.
- **Valuation Calculator**: Interactive sliders and number inputs for Area (sq ft) and Room Count with real-time prediction calculation and confidence range ($\pm 1\text{ RMSE}$).
- **Interactive Visualizations**:
  - **Area vs Price Scatter Plot**: Real-time overlay of user's prediction marker.
  - **Benchmark Comparison Bar Chart**: Visual comparison of all 4 evaluated ML candidate algorithms.
  - **Avg Price by Rooms Bar Chart**: HSL color-gradient breakdown by room count.
  - **Dataset View**: Tabular browser with live string search and filtering.
  - **Static EDA Gallery**: Embedded Matplotlib/Seaborn residual, correlation, and feature distribution charts.

---

## 4. Automated Testing & Verification

A test suite (`tests/test_app.py`) was created and executed using `pytest`. All **9 out of 9 tests passed 100%**:

```
============================= test session starts =============================
platform win32 -- Python 3.12.15, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\projects\house price prediction
collected 9 items

tests\test_app.py .........                                              [100%]

======================== 9 passed in 4.28s =========================
```

### Verified Scenarios:
1. `test_data_loader`: Correct Excel data loading and schema validation (47 rows).
2. `test_model_artifact`: Loading `.pkl` model artifact and verifying positive price predictions.
3. `test_metrics_json`: Validating presence of MAE, RMSE, R², and comparison scores.
4. `test_api_health`: `/api/health` endpoint returning `status: ok` and active model name.
5. `test_api_predict_valid`: `/api/predict` returning predicted price, price range, and formatted string.
6. `test_api_predict_invalid`: Rejecting non-positive/invalid inputs with HTTP 400 Bad Request.
7. `test_api_metrics`: `/api/metrics` serving model metrics.
8. `test_api_data`: `/api/data` serving full dataset arrays for charts.
9. `test_index_route`: `/` route serving single-page application index HTML.

---

## 5. How to Run the Application

### 1. Run the Flask Server
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Launch Flask development server
python run.py
```

### 2. Open in Browser
Navigate to **`http://localhost:5000`** in your web browser.

### 3. Run Unit & Integration Tests
```bash
.\.venv\Scripts\python.exe -m pytest tests/
```
