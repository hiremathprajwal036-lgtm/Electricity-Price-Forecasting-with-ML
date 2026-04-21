import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# data loader tests

def test_synthetic_data_shape():
    from data_loader import generate_synthetic_data
    df = generate_synthetic_data(start="2022-01-01", end="2022-12-31")
    assert len(df) > 8000
    assert "price" in df.columns
    assert "load" in df.columns
    assert "temperature" in df.columns


def test_clean_data_no_nan():
    from data_loader import generate_synthetic_data, clean_data
    df = generate_synthetic_data(start="2022-01-01", end="2022-06-30")
    df_clean = clean_data(df)
    assert df_clean.isnull().sum().sum() == 0


def test_clean_data_sorted():
    from data_loader import generate_synthetic_data, clean_data
    df = generate_synthetic_data(start="2022-01-01", end="2022-03-31")
    df_clean = clean_data(df)
    assert df_clean.index.is_monotonic_increasing


# feature tests

def test_lag_features_exist():
    from data_loader import generate_synthetic_data, clean_data
    from features import add_lag_features
    df = generate_synthetic_data(start="2022-01-01", end="2022-03-31")
    df_clean = clean_data(df)
    df_feat = add_lag_features(df_clean, lags=[1, 24])
    assert "price_lag_1h" in df_feat.columns
    assert "price_lag_24h" in df_feat.columns


def test_calendar_features_range():
    from data_loader import generate_synthetic_data, clean_data
    from features import add_calendar_features
    df = generate_synthetic_data(start="2022-01-01", end="2022-03-31")
    df_clean = clean_data(df)
    df_feat = add_calendar_features(df_clean)
    assert df_feat["hour_sin"].between(-1, 1).all()
    assert df_feat["hour_cos"].between(-1, 1).all()
    assert df_feat["is_weekend"].isin([0, 1]).all()


def test_build_features_no_nan():
    from data_loader import generate_synthetic_data, clean_data
    from features import build_features
    df = generate_synthetic_data(start="2022-01-01", end="2022-06-30")
    df_clean = clean_data(df)
    df_feat = build_features(df_clean)
    assert df_feat.isnull().sum().sum() == 0
    assert len(df_feat) > 1000


def test_temporal_split_no_leakage():
    # make sure train dates always come before val, and val before test
    from data_loader import generate_synthetic_data, clean_data
    from features import build_features, train_test_split_temporal
    df = generate_synthetic_data(start="2020-01-01", end="2022-12-31")
    df_clean = clean_data(df)
    df_feat = build_features(df_clean)
    (_, _, d_train, _, _, d_val, _, _, d_test, _) = train_test_split_temporal(df_feat)
    assert d_train[-1] < d_val[0]
    assert d_val[-1] < d_test[0]


# signal tests

def test_signal_values():
    from trading_strategy import generate_signals
    np.random.seed(0)
    prices = np.random.uniform(20, 80, 1000)
    dates = pd.date_range("2022-01-01", periods=1000, freq="h")
    sig_df = generate_signals(prices, dates)
    assert set(sig_df["signal"].unique()).issubset({-1, 0, 1})
    assert sig_df["signal_str"].isin(["BUY", "SELL", "HOLD"]).all()


def test_signal_strings_match():
    from trading_strategy import generate_signals
    prices = np.array([20.0] * 500 + [80.0] * 500)
    dates = pd.date_range("2022-01-01", periods=1000, freq="h")
    sig_df = generate_signals(prices, dates, buy_margin=0.05, sell_margin=0.05)
    assert (sig_df[sig_df["signal"] == -1]["signal_str"] == "BUY").all()
    assert (sig_df[sig_df["signal"] == 1]["signal_str"] == "SELL").all()


# backtest tests

def test_backtest_capital_positive():
    from trading_strategy import generate_signals, backtest
    np.random.seed(1)
    prices = np.random.uniform(30, 70, 500)
    dates = pd.date_range("2022-01-01", periods=500, freq="h")
    sig_df = generate_signals(prices, dates)
    result_df, trade_df = backtest(sig_df, prices, initial_capital=10_000)
    assert result_df["portfolio_value"].min() > 0


def test_backtest_portfolio_nonnegative():
    from trading_strategy import generate_signals, backtest
    np.random.seed(99)
    prices = np.concatenate([np.full(50, 5.0), np.random.uniform(40, 80, 200)])
    dates = pd.date_range("2022-01-01", periods=250, freq="h")
    sig_df = generate_signals(prices, dates, buy_margin=0.01, sell_margin=0.99)
    result_df, trade_df = backtest(sig_df, prices, initial_capital=50_000)
    assert result_df["portfolio_value"].min() >= 0


def test_metrics_keys():
    from trading_strategy import generate_signals, backtest, compute_strategy_metrics
    np.random.seed(2)
    prices = np.random.uniform(30, 70, 500)
    dates = pd.date_range("2022-01-01", periods=500, freq="h")
    sig_df = generate_signals(prices, dates)
    result_df, trade_df = backtest(sig_df, prices, initial_capital=10_000)
    metrics = compute_strategy_metrics(result_df, trade_df, initial_capital=10_000)
    for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                "n_trades", "win_rate_pct", "final_portfolio_value"]:
        assert key in metrics


# model tests

def test_random_forest_predict_shape():
    from models import RandomForestForecaster
    X_train = np.random.rand(500, 20)
    y_train = np.random.rand(500)
    X_test = np.random.rand(100, 20)
    model = RandomForestForecaster(params={"n_estimators": 10, "random_state": 42})
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (100,)


def test_random_forest_feature_importance():
    from models import RandomForestForecaster
    X_train = np.random.rand(200, 5)
    y_train = np.random.rand(200)
    feat_names = ["a", "b", "c", "d", "e"]
    model = RandomForestForecaster(params={"n_estimators": 10, "random_state": 42})
    model.fit(X_train, y_train, feature_names=feat_names)
    fi = model.feature_importance()
    assert len(fi) == 5
    assert all(v >= 0 for v in fi.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
