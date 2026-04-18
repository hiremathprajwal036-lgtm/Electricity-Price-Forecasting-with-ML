import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_signals(predicted_prices, dates, threshold_window=168,
                     buy_margin=0.08, sell_margin=0.08):
    # compare predicted price to a rolling average of the past week
    # if predicted price is 8% above average -> sell
    # if predicted price is 8% below average -> buy
    # otherwise -> hold
    df = pd.DataFrame({"predicted_price": predicted_prices}, index=dates)

    df["threshold"] = (
        df["predicted_price"].rolling(threshold_window, min_periods=24).mean().shift(1)
    )
    df["threshold"] = df["threshold"].fillna(df["predicted_price"].mean())

    buy_price = df["threshold"] * (1 - buy_margin)
    sell_price = df["threshold"] * (1 + sell_margin)

    conditions = [
        df["predicted_price"] <= buy_price,
        df["predicted_price"] >= sell_price,
    ]
    choices = [-1, 1]  # -1 = buy, +1 = sell
    df["signal"] = np.select(conditions, choices, default=0)
    df["signal_str"] = df["signal"].map({-1: "BUY", 0: "HOLD", 1: "SELL"})

    return df


def backtest(signal_df, actual_prices, capacity_mwh=10.0, efficiency=0.90,
             trade_volume_mwh=1.0, initial_capital=10_000.0, transaction_cost=0.50):
    # simulate trading on actual prices using our signals
    # models a battery that can store and release energy
    n = len(signal_df)
    signals = signal_df["signal"].values

    capital = initial_capital
    storage = 0.0
    portfolio_values = []
    trade_log = []

    for i in range(n):
        price = actual_prices[i]
        sig = signals[i]

        if sig == -1 and storage < capacity_mwh:
            # buy: charge the battery
            vol = min(trade_volume_mwh, capacity_mwh - storage)
            cost = price * vol + transaction_cost * vol
            if capital >= cost:
                capital -= cost
                storage += vol * efficiency
                trade_log.append({
                    "timestamp": signal_df.index[i],
                    "action": "BUY",
                    "price": price,
                    "volume": vol,
                    "pnl": -cost,
                    "storage": storage,
                    "capital": capital,
                })

        elif sig == 1 and storage > 0:
            # sell: discharge the battery
            vol = min(trade_volume_mwh, storage)
            revenue = price * vol - transaction_cost * vol
            capital += revenue
            storage -= vol
            trade_log.append({
                "timestamp": signal_df.index[i],
                "action": "SELL",
                "price": price,
                "volume": vol,
                "pnl": revenue,
                "storage": storage,
                "capital": capital,
            })

        # portfolio value = cash + value of stored energy at current price
        portfolio_values.append(capital + storage * price)

    result_df = signal_df.copy()
    result_df["actual_price"] = actual_prices
    result_df["portfolio_value"] = portfolio_values

    trade_df = pd.DataFrame(trade_log)
    return result_df, trade_df


def compute_strategy_metrics(result_df, trade_df, initial_capital=10_000.0):
    pv = result_df["portfolio_value"].values
    returns = pd.Series(pv).pct_change().dropna()

    total_return = (pv[-1] - initial_capital) / initial_capital * 100
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(8760)
        if returns.std() > 0 else 0.0
    )

    roll_max = pd.Series(pv).cummax()
    drawdown = (pd.Series(pv) - roll_max) / roll_max
    max_drawdown = drawdown.min() * 100

    n_trades = len(trade_df)
    if n_trades > 0 and len(trade_df.columns) > 0 and "pnl" in trade_df.columns:
        win_rate = (trade_df["pnl"] > 0).sum() / n_trades * 100
        avg_pnl = trade_df["pnl"].mean()
        total_pnl = trade_df["pnl"].sum()
    else:
        win_rate = avg_pnl = total_pnl = 0.0

    signal_counts = result_df["signal"].value_counts().to_dict()

    return {
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "n_buy_signals": signal_counts.get(-1, 0),
        "n_sell_signals": signal_counts.get(1, 0),
        "n_hold_signals": signal_counts.get(0, 0),
        "final_portfolio_value": round(pv[-1], 2),
    }


def print_strategy_report(metrics, model_name=""):
    label = f" [{model_name}]" if model_name else ""
    print(f"\n--- strategy results{label} ---")
    print(f"total return:     {metrics['total_return_pct']:+.2f}%")
    print(f"sharpe ratio:      {metrics['sharpe_ratio']:.3f}")
    print(f"max drawdown:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"final portfolio:  EUR {metrics['final_portfolio_value']:,.2f}")
    print(f"total trades:      {metrics['n_trades']}")
    print(f"win rate:          {metrics['win_rate_pct']:.1f}%")
    print(f"avg pnl/trade:    EUR {metrics['avg_pnl_per_trade']:.2f}")
    print(f"buy / sell / hold: {metrics['n_buy_signals']} / {metrics['n_sell_signals']} / {metrics['n_hold_signals']}")


def plot_strategy(result_df, trade_df, model_name="model", n_days=30):
    n_hours = n_days * 24
    subset = result_df.iloc[-n_hours:]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ax1 = axes[0]
    ax1.plot(subset.index, subset["actual_price"],
             color="black", linewidth=1, label="actual", alpha=0.8)
    ax1.plot(subset.index, subset["predicted_price"],
             color="steelblue", linewidth=1.2, label="predicted", alpha=0.9)
    ax1.plot(subset.index, subset["threshold"],
             color="gray", linewidth=1, linestyle="--", label="threshold", alpha=0.6)

    if len(trade_df) > 0:
        trade_sub = trade_df[trade_df["timestamp"] >= subset.index[0]]
        buys = trade_sub[trade_sub["action"] == "BUY"]
        sells = trade_sub[trade_sub["action"] == "SELL"]
        if len(buys):
            ax1.scatter(buys["timestamp"], buys["price"],
                        marker="^", color="green", s=60, zorder=5, label="buy", alpha=0.9)
        if len(sells):
            ax1.scatter(sells["timestamp"], sells["price"],
                        marker="v", color="red", s=60, zorder=5, label="sell", alpha=0.9)

    ax1.set_ylabel("price (EUR/MWh)")
    ax1.set_title(f"{model_name} - prices and signals (last {n_days} days)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(subset.index, subset["portfolio_value"],
             color="purple", linewidth=1.5, label="portfolio value (EUR)")
    ax2.axhline(subset["portfolio_value"].iloc[0], color="gray",
                linestyle="--", linewidth=1, alpha=0.6)
    ax2.fill_between(subset.index, subset["portfolio_value"].iloc[0],
                     subset["portfolio_value"], alpha=0.15, color="purple")
    ax2.set_ylabel("portfolio value (EUR)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for sig, color in [(-1, "green"), (0, "lightgray"), (1, "red")]:
        mask = subset["signal"] == sig
        label = {-1: "buy", 0: "hold", 1: "sell"}[sig]
        ax3.fill_between(subset.index, 0, mask.astype(int),
                         color=color, alpha=0.6, label=label)
    ax3.set_ylabel("signal")
    ax3.set_xlabel("date")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["", "active"])
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"strategy_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"saved plot -> {path}")


def plot_portfolio_comparison(portfolio_series):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["purple", "steelblue", "darkorange", "forestgreen"]
    for (name, series), color in zip(portfolio_series.items(), colors):
        normalized = series / series.iloc[0] * 100
        ax.plot(series.index, normalized, label=name, linewidth=1.5, color=color)
    ax.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.5, label="start")
    ax.set_title("portfolio comparison across models (indexed to 100)")
    ax.set_ylabel("portfolio value (indexed)")
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "portfolio_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"saved -> {path}")


def run_strategy(predictions_csv=None):
    print("\n--- running trading strategy ---")

    if predictions_csv is None:
        predictions_csv = os.path.join(RESULTS_DIR, "test_predictions.csv")

    if not os.path.exists(predictions_csv):
        print("no predictions file found, running training first...")
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from train import run_training
        run_training()

    df = pd.read_csv(predictions_csv, index_col=0, parse_dates=True)
    actual = df["actual"].values

    model_cols = [c for c in df.columns if c.startswith("pred_")]
    all_metrics = {}
    portfolio_series = {}

    for col in model_cols:
        model_name = col.replace("pred_", "").replace("_", " ")
        preds = df[col].dropna().values
        valid_dates = df[col].dropna().index

        print(f"\nrunning strategy for: {model_name}")

        signal_df = generate_signals(preds, valid_dates)
        actual_aligned = df.loc[valid_dates, "actual"].values
        result_df, trade_df = backtest(signal_df, actual_aligned)

        metrics = compute_strategy_metrics(result_df, trade_df)
        print_strategy_report(metrics, model_name)
        all_metrics[model_name] = metrics

        if len(trade_df) > 0:
            trade_path = os.path.join(RESULTS_DIR, f"trades_{model_name.replace(' ', '_')}.csv")
            trade_df.to_csv(trade_path, index=False)

        plot_strategy(result_df, trade_df, model_name=model_name)
        portfolio_series[model_name] = result_df["portfolio_value"]

    if len(portfolio_series) > 1:
        plot_portfolio_comparison(portfolio_series)

    metrics_path = os.path.join(RESULTS_DIR, "strategy_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nsaved strategy metrics -> {metrics_path}")
    print("trading strategy done")
    return all_metrics


if __name__ == "__main__":
    run_strategy()
