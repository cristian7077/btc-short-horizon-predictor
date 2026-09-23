import numpy as np
import pandas as pd


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def make_features(df):
    x = df.copy()

    # Multi-horizon returns and ranges.
    for n in [1, 2, 3, 5, 10, 15, 30, 60]:
        x[f"ret_{n}"] = x["close"].pct_change(n)
        x[f"range_{n}"] = (
            x["high"].rolling(n).max() - x["low"].rolling(n).min()
        ) / x["close"]

    # Moving averages.
    for n in [5, 9, 13, 21, 34, 50, 100, 200]:
        ema = x["close"].ewm(span=n, adjust=False).mean()
        x[f"ema_{n}"] = ema
        x[f"dist_ema_{n}"] = x["close"] / ema - 1

    # RSI.
    for n in [7, 14, 21]:
        x[f"rsi_{n}"] = rsi(x["close"], n)

    # MACD.
    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    # Bollinger Bands.
    mid = x["close"].rolling(20).mean()
    std = x["close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    x["bb_mid"] = mid
    x["bb_upper"] = upper
    x["bb_lower"] = lower
    x["bb_width"] = (upper - lower) / mid
    x["bb_position"] = (x["close"] - lower) / (upper - lower)

    # Volatility.
    x["atr_14"] = atr(x, 14)
    x["atr_pct"] = x["atr_14"] / x["close"]
    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_30"] = x["ret_1"].rolling(30).std()
    x["vol_60"] = x["ret_1"].rolling(60).std()

    # Volume.
    for n in [5, 10, 20, 50]:
        avg = x["volume"].rolling(n).mean()
        sd = x["volume"].rolling(n).std()
        x[f"vol_ratio_{n}"] = x["volume"] / avg
        x[f"vol_z_{n}"] = (x["volume"] - avg) / sd.replace(0, np.nan)

    # Candle anatomy.
    x["body_pct"] = (x["close"] - x["open"]) / x["open"]
    x["upper_wick_pct"] = (
        x["high"] - x[["open", "close"]].max(axis=1)
    ) / x["open"]
    x["lower_wick_pct"] = (
        x[["open", "close"]].min(axis=1) - x["low"]
    ) / x["open"]

    # Rolling VWAP rather than lifetime VWAP.
    typical = (x["high"] + x["low"] + x["close"]) / 3
    pv = typical * x["volume"]
    x["vwap_60"] = pv.rolling(60).sum() / x["volume"].rolling(60).sum()
    x["dist_vwap"] = x["close"] / x["vwap_60"] - 1

    # Stochastic.
    low14 = x["low"].rolling(14).min()
    high14 = x["high"].rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    x["stoch_k"] = 100 * (x["close"] - low14) / denom
    x["stoch_d"] = x["stoch_k"].rolling(3).mean()

    # OBV and its rate of change.
    direction = np.sign(x["close"].diff()).fillna(0)
    x["obv"] = (direction * x["volume"]).cumsum()
    x["obv_change_10"] = x["obv"].diff(10)

    # Time-of-day.
    minutes = x.index.hour * 60 + x.index.minute
    x["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    x["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)

    return x.replace([np.inf, -np.inf], np.nan)


def add_targets(df, horizons=(1, 5, 15)):
    out = df.copy()
    for h in horizons:
        future = out["close"].shift(-h)
        out[f"target_{h}m"] = (future > out["close"]).astype(int)
        out[f"future_ret_{h}m"] = future / out["close"] - 1
    return out


def feature_columns(df):
    excluded = {
        "open", "high", "low", "close", "volume",
        "timestamp", "target_1m", "target_5m", "target_15m",
        "future_ret_1m", "future_ret_5m", "future_ret_15m",
    }
    return [
        c for c in df.columns
        if c not in excluded and not c.startswith("target_")
        and not c.startswith("future_ret_")
    ]
