import asyncio
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from data import get_spot_klines, get_product_book, get_recent_trades, get_ticker, orderbook_features, trade_flow_features
from features import make_features
from config import HORIZONS

app = FastAPI(title="BTC Live Lab")

MODELS = {}
for h in HORIZONS:
    p = Path("models") / f"btc_{h}m.joblib"
    if p.exists():
        MODELS[h] = joblib.load(p)

HTML = Path(__file__).with_name("web.html").read_text(encoding="utf-8")


def snapshot():
    candles = get_spot_klines(limit=300)
    features = make_features(candles)
    latest = features.iloc[-1]
    ticker = get_ticker()
    book = orderbook_features(get_product_book())
    flow = trade_flow_features(get_recent_trades())

    predictions = {}
    for h, bundle in MODELS.items():
        X = latest[bundle["features"]].to_frame().T
        if X.isna().any(axis=None):
            continue
        p = float(bundle["model"].predict_proba(X)[0, 1])
        predictions[str(h)] = {
            "up": p,
            "down": 1 - p,
            "signal": "UP" if p >= 0.5 else "DOWN",
        }

    rows = []
    for ts, row in candles.tail(180).iterrows():
        rows.append({
            "t": int(ts.timestamp() * 1000),
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": float(row["volume"]),
        })

    return {
        "price": float(ticker["price"]),
        "time": candles.index[-1].isoformat(),
        "candles": rows,
        "rsi": float(latest["rsi_14"]),
        "volume_ratio": float(latest["vol_ratio_20"]),
        "vwap_distance": float(latest["dist_vwap"]),
        "book_imbalance": book.get("book_imbalance"),
        "spread_bps": book.get("spread_bps"),
        "trade_flow": flow.get("trade_flow_imbalance"),
        "trade_volume": flow.get("recent_trade_volume"),
        "predictions": predictions,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/snapshot")
def api_snapshot():
    return snapshot()
