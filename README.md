# BTC Short-Horizon Prediction Lab — Fixed V1

This version fixes the first-run problems in the original project.

## What changed

- Uses Coinbase Exchange public BTC-USD market data instead of the Binance endpoint that returned HTTP 451 in the original setup.
- Downloads historical 1-minute candles in multiple 300-candle chunks instead of requesting only one small window.
- Collects 10,000 1-minute candles by default.
- Keeps a chronological train/test split.
- Uses RSI, multiple EMAs, MACD, Bollinger Bands, ATR, volatility, VWAP, stochastic, OBV, momentum, volume and candle-structure features.
- Includes live Coinbase order-book imbalance, spread and recent trade-flow data.
- Does not pretend Coinbase spot data provides futures funding/open interest; those fields were removed from the live layer rather than being faked.
- Saves the dataset, trained models, test metrics and feature importance.
- Includes a Streamlit dashboard.

## Run

You already have the virtual environment. From this folder:

    source .venv/bin/activate

Then:

    pip install -r requirements.txt

Train:

    python train.py

After training:

    streamlit run app.py

For terminal live output:

    python live.py

## What to expect

The first training run will make multiple public API requests because Coinbase limits candle responses per request. It may take a little while.

At the end you should have:

    data/btc_features.csv
    models/btc_1m.joblib
    models/btc_5m.joblib
    models/btc_15m.joblib
    models/metrics.json
    models/importance_1m.csv
    models/importance_5m.csv
    models/importance_15m.csv

## Important research limitation

This is NOT yet a validated profitable trading strategy.

The first model uses historical candle-derived features. Live order-book/trade-flow data are displayed but are not fed into the model because doing that correctly requires synchronized historical microstructure data.

The next major version should collect that data continuously, build a larger database, use walk-forward validation, calibrate probabilities and include fees/slippage in a paper-trading backtest.
