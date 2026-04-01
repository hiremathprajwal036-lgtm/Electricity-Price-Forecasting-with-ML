import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    start = time.time()

    print("electricity price forecasting + trading strategy")
    print("=" * 50)

    # step 1 and 2: get data and build features
    print("\nstep 1: loading data and building features")
    from src.data_loader import load_raw_data, clean_data
    from src.features import build_features

    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    features_path = os.path.join(PROCESSED_DIR, "electricity_features.csv")

    if not os.path.exists(features_path):
        df_raw = load_raw_data(use_synthetic=True)
        df_clean = clean_data(df_raw)
        df_feat = build_features(df_clean)
        df_feat.to_csv(features_path)
        print(f"features saved -> {features_path}")
    else:
        print(f"found existing features file")

    # step 3 and 4: train models and evaluate
    print("\nstep 2: training models")
    from src.train import run_training
    models_preds, y_test, dates_test, metrics = run_training()

    # step 5: trading strategy backtest
    print("\nstep 3: running trading strategy")
    from src.trading_strategy import run_strategy
    strategy_metrics = run_strategy()

    elapsed = time.time() - start
    print(f"\ndone in {elapsed:.1f}s")
    print("\noutput files:")
    print("  results/predictions.png")
    print("  results/portfolio_comparison.png")
    print("  results/feature_importance_*.png")
    print("  results/strategy_*.png")
    print("  results/metrics.json")
    print("  results/strategy_metrics.json")
    print("  results/test_predictions.csv")
    print("  results/trades_*.csv")

    print("\nmodel results:")
    for model, m in metrics.items():
        print(f"  {model:<25} MAE={m['MAE']:.3f}  R2={m['R2']:.4f}")

    print("\ntrading results:")
    for model, m in strategy_metrics.items():
        print(f"  {model:<25} return={m['total_return_pct']:+.2f}%  sharpe={m['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()
