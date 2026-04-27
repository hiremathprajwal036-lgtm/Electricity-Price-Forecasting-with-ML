# Electricity Price Forecasting + Trading Strategy

I'm a Master's student in Energy Engineering.
This project came out of my interest in flexibility markets and VPP optimization
after working on MILP-based energy dispatch modeling. The ML forecasting angle
was something I wanted to explore on top of the optimization work.

I built it to learn how time series forecasting works and how you can combine it with simple trading logic.

---

## What it does

1. Loads hourly electricity price data (synthetic data by default, but you can plug in real data from ENTSO-E)
2. Creates features from that data like price lags, rolling averages, and time of day
3. Trains XGBoost and Random Forest models to predict future prices
4. Uses those predictions to decide: buy now (cheap), sell now (expensive), or do nothing
5. Backtests that strategy and shows how much you would have made

---

## Folder structure

```
electricity-price-forecasting/
├── data/
│   ├── raw/               <- raw data goes here
│   └── processed/         <- cleaned data and features
├── src/
│   ├── data_loader.py     <- loads and cleans data
│   ├── features.py        <- creates features for the model
│   ├── models.py          <- XGBoost, Random Forest, LSTM
│   ├── train.py           <- trains the models
│   ├── predict.py         <- runs predictions on new data
│   └── trading_strategy.py  <- the buy/sell/hold logic
├── models/                <- saved model files
├── results/               <- output charts and CSVs
├── tests/                 <- unit tests
├── main.py                <- run everything from here
└── requirements.txt
```

---

## How to run it

Clone the repo and install requirements:

```bash
git clone https://github.com/hiremathprajwal036-lgtm/Electricity-Price-Forecasting-with-ML
cd electricity-price-forecasting
pip install -r requirements.txt
```

Then just run:

```bash
python main.py
```

That will do everything - load data, train models, run the trading strategy, and save charts to the `results/` folder.

If you want to run steps one at a time:

```bash
python src/data_loader.py       # load and clean data
python src/train.py             # train models
python src/predict.py           # generate predictions
python src/trading_strategy.py  # run trading backtest
```

---

## Models

I tried three models:

- **XGBoost** - works really well on tabular data, trains fast
- **Random Forest** - similar to XGBoost, good for checking which features matter most
- **LSTM** - deep learning model, needs TensorFlow installed separately

XGBoost ended up being the most accurate in my tests.

---

## Trading strategy

The idea is simple. Compare the predicted price against a rolling average of recent prices:

- if predicted price is much higher than average -> SELL
- if predicted price is much lower than average -> BUY
- otherwise -> HOLD

It simulates a 10 MWh battery that can charge (buy) and discharge (sell) energy.

---

## Results I got

```
XGBoost        MAE: 0.42   RMSE: 1.37   R2: 0.99
Random Forest  MAE: 1.45   RMSE: 2.09   R2: 0.97

XGBoost strategy:      +170.9% return, Sharpe 3.85, max drawdown -8.2%
Random Forest strategy: +169.2% return, Sharpe 3.75, max drawdown -8.2%
```

Note: these numbers are on synthetic data so they look very clean. Real market data will be noisier.

---

## Running tests

```bash
python -m pytest tests/ -v
```

14 tests, all passing.

---

## Data source

By default the project generates fake but realistic price data so you can run everything without downloading anything. If you want real data, grab it from [ENTSO-E](https://transparency.entsoe.eu/) and put it in `data/raw/`.

---

