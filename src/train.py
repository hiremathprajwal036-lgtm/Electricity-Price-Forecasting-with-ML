import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_raw_data, clean_data
from features import build_features, train_test_split_temporal
from models import (
    XGBoostForecaster,
    RandomForestForecaster,
    compute_metrics,
    XGBOOST_AVAILABLE,
    TENSORFLOW_AVAILABLE,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def prepare_data():
    features_path = os.path.join(PROCESSED_DIR, "electricity_features.csv")

    if os.path.exists(features_path):
        print("loading cached features...")
        df = pd.read_csv(features_path, index_col="timestamp", parse_dates=True)
    else:
        print("building features from scratch...")
        df_raw = load_raw_data(use_synthetic=True)
        df_clean = clean_data(df_raw)
        df = build_features(df_clean)
        df.to_csv(features_path)

    splits = train_test_split_temporal(df, target="price")
    return df, splits


def train_xgboost(X_train, y_train, X_val, y_val, feature_cols):
    print("\ntraining xgboost...")
    model = XGBoostForecaster()
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    model.save("xgboost")
    return model


def train_random_forest(X_train, y_train, feature_cols):
    print("\ntraining random forest...")
    model = RandomForestForecaster()
    model.fit(X_train, y_train, feature_names=feature_cols)
    model.save("random_forest")
    return model


def train_lstm(X_train, y_train, X_val, y_val):
    if not TENSORFLOW_AVAILABLE:
        print("skipping LSTM (tensorflow not installed)")
        return None
    print("\ntraining LSTM...")
    from models import LSTMForecaster
    model = LSTMForecaster(sequence_len=24, units=64)
    model.fit(X_train, y_train, X_val, y_val, epochs=20)
    model.save("lstm")  # type: ignore
    return model


def evaluate_all(models_preds, y_test):
    print("\n--- test set results ---")
    all_metrics = {}
    for name, preds in models_preds.items():
        valid_mask = ~np.isnan(preds)
        metrics = compute_metrics(y_test[valid_mask], preds[valid_mask], name)
        all_metrics[name] = metrics
    return all_metrics


def plot_predictions(dates_test, y_test, models_preds, n_days=7):
    n_hours = n_days * 24
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(dates_test[-n_hours:], y_test[-n_hours:],
            label="actual", color="black", linewidth=1.5, zorder=5)
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
    for (name, preds), color in zip(models_preds.items(), colors):
        valid = ~np.isnan(preds[-n_hours:])
        if valid.any():
            ax.plot(dates_test[-n_hours:][valid], preds[-n_hours:][valid],
                    label=name, alpha=0.8, linewidth=1.2, color=color)
    ax.set_title(f"predictions vs actual - last {n_days} days")
    ax.set_xlabel("date")
    ax.set_ylabel("price (EUR/MWh)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "predictions.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"saved plot -> {path}")


def plot_feature_importance(model, top_n=20, model_name="xgboost"):
    fi = model.feature_importance()
    if not fi:
        return
    top = dict(list(fi.items())[:top_n])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(top.keys())[::-1], list(top.values())[::-1], color="steelblue")
    ax.set_title(f"{model_name} - top {top_n} features")
    ax.set_xlabel("importance")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"saved -> {path}")


def save_metrics(metrics):
    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved metrics -> {path}")


def save_test_predictions(dates_test, y_test, models_preds):
    df = pd.DataFrame({"actual": y_test}, index=dates_test)
    for name, preds in models_preds.items():
        df[f"pred_{name.lower().replace(' ', '_')}"] = preds
    path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    df.to_csv(path)
    print(f"saved predictions -> {path}")


def run_training():
    print("\n--- training models ---\n")

    df, splits = prepare_data()
    (
        X_train, y_train, dates_train,
        X_val, y_val, dates_val,
        X_test, y_test, dates_test,
        feature_cols,
    ) = splits

    models_preds = {}

    # naive baseline: just use yesterday's price
    naive_col = "price_lag_24h"
    if naive_col in df.columns:
        models_preds["naive_24h_lag"] = df.loc[dates_test, naive_col].values

    if XGBOOST_AVAILABLE:
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)
        models_preds["xgboost"] = xgb_model.predict(X_test)
        plot_feature_importance(xgb_model, model_name="xgboost")

    rf_model = train_random_forest(X_train, y_train, feature_cols)
    models_preds["random_forest"] = rf_model.predict(X_test)
    plot_feature_importance(rf_model, model_name="random_forest")

    lstm_model = train_lstm(X_train, y_train, X_val, y_val)
    if lstm_model is not None:
        models_preds["lstm"] = lstm_model.predict(X_test)

    metrics = evaluate_all(models_preds, y_test)

    save_metrics(metrics)
    plot_predictions(dates_test, y_test, models_preds)
    save_test_predictions(dates_test, y_test, models_preds)

    print("\ntraining done. results in results/")
    return models_preds, y_test, dates_test, metrics


if __name__ == "__main__":
    run_training()
