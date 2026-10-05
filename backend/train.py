"""
train.py
Trains and compares regression models on house price data, evaluates them with cross-validation,
saves the best model artifact, and generates EDA & performance plots.
"""
import os
import sys
import json

# Set Matplotlib config dir to local workspace directory before importing matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".matplotlib_cache")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from data_loader import load_data

MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
STATIC_DIR = os.path.join(BASE_DIR, "..", "frontend", "static")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def create_candidate_pipelines():
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(n_estimators=50, random_state=42))
        ])
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2), "r2": round(float(r2), 4), "y_pred": y_pred.tolist()}


def plot_eda(df: pd.DataFrame):
    """Save clean EDA charts to frontend/static/."""
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({'figure.facecolor': '#111827', 'axes.facecolor': '#1f2937', 'text.color': '#f9fafb', 'axes.labelcolor': '#f9fafb', 'xtick.color': '#d1d5db', 'ytick.color': '#d1d5db'})

    # 1. Scatter: area vs price
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["area"], df["price"] / 1000, alpha=0.8, color="#38bdf8", edgecolors="#0284c7", s=60)
    ax.set_xlabel("Area (sq ft)", fontsize=11)
    ax.set_ylabel("Price ($k)", fontsize=11)
    ax.set_title("Area vs Price", fontsize=13, fontweight="bold", color="#f9fafb")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "area_vs_price.png"), dpi=120)
    plt.close(fig)

    # 2. Box: rooms vs price
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="price", by="rooms", ax=ax, patch_artist=True,
               boxprops=dict(facecolor="#1e3a8a", color="#60a5fa"),
               medianprops=dict(color="#f43f5e", linewidth=2.5))
    ax.set_xlabel("Rooms", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_title("Price Distribution by Rooms", fontsize=13, fontweight="bold", color="#f9fafb")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "rooms_vs_price.png"), dpi=120)
    plt.close(fig)

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    corr = df[["area", "rooms", "price"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako", ax=ax, linewidths=0.5, cbar=False)
    ax.set_title("Correlation Matrix", fontsize=12, fontweight="bold", color="#f9fafb")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "correlation.png"), dpi=120)
    plt.close(fig)


def plot_residuals_and_comparison(y_test, y_pred, model_results, best_name):
    """Save model performance and comparison charts."""
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({'figure.facecolor': '#111827', 'axes.facecolor': '#1f2937', 'text.color': '#f9fafb', 'axes.labelcolor': '#f9fafb', 'xtick.color': '#d1d5db', 'ytick.color': '#d1d5db'})

    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(np.array(y_test) / 1000, np.array(y_pred) / 1000,
                    alpha=0.85, color="#818cf8", edgecolors="#4f46e5", s=60)
    mn = min(min(y_test), min(y_pred)) / 1000
    mx = max(max(y_test), max(y_pred)) / 1000
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.8, label="Ideal fit")
    axes[0].set_xlabel("Actual Price ($k)", fontsize=11)
    axes[0].set_ylabel("Predicted Price ($k)", fontsize=11)
    axes[0].set_title(f"Actual vs Predicted ({best_name})", fontsize=12, fontweight="bold", color="#f9fafb")
    axes[0].legend(facecolor="#111827", edgecolor="#374151", labelcolor="#f9fafb")

    # Residuals
    residuals = np.array(y_test) - np.array(y_pred)
    axes[1].scatter(np.array(y_pred) / 1000, residuals / 1000, alpha=0.85,
                    color="#f472b6", edgecolors="#db2777", s=60)
    axes[1].axhline(0, color="#ef4444", linestyle="--", linewidth=1.8)
    axes[1].set_xlabel("Predicted Price ($k)", fontsize=11)
    axes[1].set_ylabel("Residual ($k)", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold", color="#f9fafb")

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "model_performance.png"), dpi=120)
    plt.close(fig)

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(model_results.keys())
    scores = [res["cv_r2_mean"] * 100 for res in model_results.values()]
    colors = ["#34d399" if n == best_name else "#60a5fa" for n in names]

    bars = ax.bar(names, scores, color=colors, width=0.55)
    ax.set_ylabel("5-Fold CV R² Score (%)", fontsize=11)
    ax.set_title("Model Comparison (CV R²)", fontsize=12, fontweight="bold", color="#f9fafb")
    ax.set_ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', color='#f9fafb', fontweight='bold', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "model_comparison.png"), dpi=120)
    plt.close(fig)


def train():
    df = load_data()
    print(f"Dataset: {len(df)} samples, columns: {list(df.columns)}")

    plot_eda(df)
    print("EDA plots saved.")

    X = df[["area", "rooms"]].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = create_candidate_pipelines()
    comparison_results = {}
    best_name = None
    best_cv_score = -float("inf")
    best_pipeline = None

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
        eval_metrics = evaluate(pipeline, X_test, y_test)
        eval_metrics.pop("y_pred", None)

        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        comparison_results[name] = {
            "mae": eval_metrics["mae"],
            "rmse": eval_metrics["rmse"],
            "r2": eval_metrics["r2"],
            "cv_r2_mean": round(cv_mean, 4),
            "cv_r2_std": round(cv_std, 4),
        }

        print(f"[{name}] CV R2: {cv_mean:.4f} +/- {cv_std:.4f} | Test R2: {eval_metrics['r2']} | MAE: ${eval_metrics['mae']:,.0f}")

        if cv_mean > best_cv_score:
            best_cv_score = cv_mean
            best_name = name
            best_pipeline = pipeline

    print(f"\nBest Model Selected: {best_name} (CV R2 = {best_cv_score:.4f})")

    # Evaluate best model on test set
    best_eval = evaluate(best_pipeline, X_test, y_test)
    y_pred_list = best_eval.pop("y_pred")

    plot_residuals_and_comparison(y_test.tolist(), y_pred_list, comparison_results, best_name)
    print("Performance & comparison plots saved.")

    final_metrics = {
        "model_name": best_name,
        "mae": best_eval["mae"],
        "rmse": best_eval["rmse"],
        "r2": best_eval["r2"],
        "cv_r2_mean": comparison_results[best_name]["cv_r2_mean"],
        "cv_r2_std": comparison_results[best_name]["cv_r2_std"],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "features": ["area", "rooms"],
        "model_comparison": comparison_results
    }

    joblib.dump(best_pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Model saved -> {MODEL_PATH}")
    print(f"Metrics: MAE=${final_metrics['mae']:,.0f}  RMSE=${final_metrics['rmse']:,.0f}  R2={final_metrics['r2']}")
    return best_pipeline, final_metrics


if __name__ == "__main__":
    train()

