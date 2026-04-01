import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def generate_synthetic_data(start="2019-01-01", end="2023-12-31", freq="h", seed=42):
    # makes fake hourly electricity prices that behave like real ones
    # useful for testing without needing to download actual market data
    np.random.seed(seed)
    idx = pd.date_range(start=start, end=end, freq=freq)
    n = len(idx)

    trend = np.linspace(40, 60, n)

    # prices go up during the day and down at night
    hour_effect = 10 * np.sin(2 * np.pi * (idx.hour - 6) / 24)

    # weekends are usually cheaper
    day_effect = np.where(idx.dayofweek < 5, 5, -5)

    # winter prices tend to be higher
    month_effect = -8 * np.cos(2 * np.pi * idx.month / 12)

    noise = np.random.normal(0, 3, n)

    # occasional price spikes like in real markets
    spikes = np.zeros(n)
    spike_idx = np.random.choice(n, size=int(n * 0.005), replace=False)
    spikes[spike_idx] = np.random.uniform(50, 150, len(spike_idx))

    price = trend + hour_effect + day_effect + month_effect + noise + spikes
    price = np.clip(price, -10, 300)

    load = (
        35000
        + 5000 * np.sin(2 * np.pi * (idx.hour - 6) / 24)
        + np.where(idx.dayofweek < 5, 3000, -2000)
        + np.random.normal(0, 500, n)
    )

    temperature = (
        15
        - 10 * np.cos(2 * np.pi * idx.dayofyear / 365)
        + np.random.normal(0, 2, n)
    )

    wind = np.abs(np.random.normal(8000, 3000, n))

    df = pd.DataFrame({
        "timestamp": idx,
        "price": price,
        "load": load,
        "temperature": temperature,
        "wind_generation": wind,
    })
    df.set_index("timestamp", inplace=True)
    return df


def load_raw_data(use_synthetic=True):
    ensure_dirs()
    cache_path = os.path.join(RAW_DIR, "electricity_raw.csv")

    if os.path.exists(cache_path):
        print(f"loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col="timestamp", parse_dates=True)
        return df

    if use_synthetic:
        print("generating synthetic electricity price data...")
        df = generate_synthetic_data()
        df.to_csv(cache_path)
        print(f"saved to {cache_path} ({len(df):,} rows)")
        return df

    raise FileNotFoundError("no raw data found. set use_synthetic=True or add data to data/raw/")


def clean_data(df):
    original_len = len(df)

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df.interpolate(method="time", limit=3)
    df = df.dropna()

    # remove extreme outliers using IQR
    Q1 = df["price"].quantile(0.01)
    Q3 = df["price"].quantile(0.99)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outlier_mask = (df["price"] >= lower) & (df["price"] <= upper)
    removed = (~outlier_mask).sum()
    df = df[outlier_mask]

    print(f"cleaned: {original_len:,} -> {len(df):,} rows ({removed} outliers removed)")
    return df


def save_processed(df, filename="electricity_processed.csv"):
    ensure_dirs()
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path)
    print(f"saved processed data -> {path}")
    return path


def load_processed(filename="electricity_features.csv"):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}. run feature engineering first.")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    print(f"loaded: {len(df):,} rows, {df.shape[1]} features")
    return df


if __name__ == "__main__":
    df_raw = load_raw_data(use_synthetic=True)
    df_clean = clean_data(df_raw)
    save_processed(df_clean, "electricity_processed.csv")
    print(df_clean.head())
    print(df_clean["price"].describe())
