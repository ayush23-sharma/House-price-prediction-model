"""
data_loader.py
Loads house price data from the xlsx file and returns a clean DataFrame.
"""
import os
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "house_prices.xlsx")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read house_prices.xlsx and return a cleaned DataFrame."""
    df = pd.read_excel(path, sheet_name="house_prices")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna()
    df = df[(df["area"] > 0) & (df["rooms"] > 0) & (df["price"] > 0)]
    df = df.astype({"area": float, "rooms": int, "price": float})
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(df.describe())
