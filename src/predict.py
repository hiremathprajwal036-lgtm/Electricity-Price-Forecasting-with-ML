import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model not found: {path} - run train.py first")
    return joblib.load(path)


def run_predictions():
    from features import build_features, train_test_split_temporal
    from data_loader import load_raw_data, clean_data

    print("loading data...")
    df_raw = load_raw_data(use_synthetic=True)
    df_clean = clean_data(df_raw)
    df_feat = build_features(df_clean)

    splits = train_test_split_temporal(df_feat, target="price")
    (_, _, _, _, _, _, X_test, y_test, dates_test, _) = splits

    predictions = {"actual": y_test}

    for model_name in ["xgboost", "random_forest"]:
        path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        if os.path.exists(path):
            print(f"predicting with {model_name}...")
            model = load_model(model_name)
            predictions[f"pred_{model_name}"] = model.predict(X_test)

    result_df = pd.DataFrame(predictions, index=dates_test)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    result_df.to_csv(out_path)
    print(f"predictions saved -> {out_path}")
    return result_df


if __name__ == "__main__":
    run_predictions()
