import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "processed"
)


def add_lag_features(df, target="price", lags=None):
    # lag features = past prices as inputs to the model
    # e.g. price_lag_24h means "what was the price 24 hours ago"
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lags:
        df[f"{target}_lag_{lag}h"] = df[target].shift(lag)
    return df


def add_rolling_features(df, target="price", windows=None):
    # rolling stats over the past N hours
    if windows is None:
        windows = [6, 24, 48, 168]
    for w in windows:
        df[f"{target}_roll_mean_{w}h"] = df[target].shift(1).rolling(w).mean()
        df[f"{target}_roll_std_{w}h"] = df[target].shift(1).rolling(w).std()
        df[f"{target}_roll_min_{w}h"] = df[target].shift(1).rolling(w).min()
        df[f"{target}_roll_max_{w}h"] = df[target].shift(1).rolling(w).max()
    return df


def add_calendar_features(df):
    # time-based features - hour, day of week, month etc.
    # using sin/cos so the model understands that hour 23 is close to hour 0
    idx = df.index

    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)

    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = ((idx.hour >= 8) & (idx.hour <= 20)).astype(int)
    df["is_night"] = ((idx.hour >= 22) | (idx.hour <= 6)).astype(int)

    df["dayofyear_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)

    return df


def add_momentum_features(df, target="price"):
    df[f"{target}_change_1h"] = df[target].diff(1)
    df[f"{target}_change_24h"] = df[target].diff(24)
    df[f"{target}_change_168h"] = df[target].diff(168)
    df[f"{target}_pct_change_24h"] = df[target].pct_change(24)
    df[f"{target}_vs_daily_mean"] = (
        df[target] - df[target].shift(1).rolling(24).mean()
    )
    return df


def add_external_features(df):
    # add lagged versions of load, temperature, and wind
    for col in ["load", "temperature", "wind_generation"]:
        if col in df.columns:
            df[f"{col}_lag_24h"] = df[col].shift(24)
            df[f"{col}_lag_48h"] = df[col].shift(48)
            df[f"{col}_roll_mean_24h"] = df[col].shift(1).rolling(24).mean()
    return df


def build_features(df, target="price"):
    print("building features...")
    df = df.copy()

    df = add_lag_features(df, target)
    df = add_rolling_features(df, target)
    df = add_calendar_features(df)
    df = add_momentum_features(df, target)
    df = add_external_features(df)

    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"dropped {before - after} rows with NaN -> {after:,} rows")
    print(f"total features: {df.shape[1]} (including target)")
    return df


def get_feature_columns(df, target="price"):
    return [c for c in df.columns if c != target]


def train_test_split_temporal(df, target="price", test_ratio=0.15, val_ratio=0.10):
    # split in order - no shuffling, because we can't use future data to predict the past
    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size

    feature_cols = get_feature_columns(df, target)

    X = df[feature_cols].values
    y = df[target].values

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size: train_size + val_size]
    y_val = y[train_size: train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    dates_train = df.index[:train_size]
    dates_val = df.index[train_size: train_size + val_size]
    dates_test = df.index[train_size + val_size:]

    print(f"train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")
    return (
        X_train, y_train, dates_train,
        X_val, y_val, dates_val,
        X_test, y_test, dates_test,
        feature_cols,
    )


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_data

    df_raw = load_raw_data(use_synthetic=True)
    df_clean = clean_data(df_raw)
    df_feat = build_features(df_clean)

    path = os.path.join(PROCESSED_DIR, "electricity_features.csv")
    df_feat.to_csv(path)
    print(f"saved features to {path}")
    print(df_feat.head(3))
