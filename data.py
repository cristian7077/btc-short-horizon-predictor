import time
import requests
import pandas as pd
import numpy as np

from config import (
    SYMBOL,
    INTERVAL,
    COINBASE_CANDLE_LIMIT,
    HISTORY_CANDLES,
)

BASE_URL = "https://api.exchange.coinbase.com"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "BTC-Short-Horizon-Research/1.0",
    "Accept": "application/json",
})


def _request(url, params=None, retries=4):
    last_error = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_error


def get_spot_klines(
    symbol=SYMBOL,
    interval=INTERVAL,
    limit=HISTORY_CANDLES,
):
    """Download historical candles in Coinbase-sized chunks."""
    seconds_map = {"1m": 60, "5m": 300, "15m": 900}
    if interval not in seconds_map:
        raise ValueError("Supported intervals: 1m, 5m, 15m")

    step = seconds_map[interval]
    remaining = int(limit)
    end = pd.Timestamp.now(tz="UTC").floor("min")
    frames = []
    total = 0

    while remaining > 0:
        batch_size = min(remaining, COINBASE_CANDLE_LIMIT)
        # Leave a tiny overlap; duplicates are removed later.
        start = end - pd.Timedelta(seconds=step * batch_size)

        url = f"{BASE_URL}/products/{symbol}/candles"
        params = {
            "granularity": step,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

        raw = _request(url, params=params)
        if not raw:
            break

        batch = pd.DataFrame(
            raw,
            columns=["timestamp", "low", "high", "open", "close", "volume"],
        )
        frames.append(batch)
        total += len(batch)

        oldest = min(int(row[0]) for row in raw)
        end = pd.to_datetime(oldest, unit="s", utc=True) - pd.Timedelta(seconds=step)
        remaining -= len(batch)

        print(f"\rDownloading candles: {min(total, limit)}/{limit}", end="", flush=True)

        # Be polite to the public API.
        time.sleep(0.12)

        # Safety against a provider returning the same window repeatedly.
        if len(raw) == 0:
            break

    print()

    if not frames:
        raise RuntimeError("Coinbase returned no candle data.")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna()
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    # Keep exactly the requested number of rows at most.
    return df.tail(limit)


def get_ticker(symbol=SYMBOL):
    return _request(f"{BASE_URL}/products/{symbol}/ticker")


def get_product_book(symbol=SYMBOL, level=2):
    return _request(
        f"{BASE_URL}/products/{symbol}/book",
        params={"level": level},
    )


def get_recent_trades(symbol=SYMBOL):
    return _request(f"{BASE_URL}/products/{symbol}/trades")


def orderbook_features(book):
    bids = [(float(p), float(q)) for p, q, *_ in book.get("bids", [])]
    asks = [(float(p), float(q)) for p, q, *_ in book.get("asks", [])]

    if not bids or not asks:
        return {}

    bid_depth = sum(q for _, q in bids)
    ask_depth = sum(q for _, q in asks)
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2

    return {
        "book_imbalance": (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-12),
        "spread_bps": (best_ask - best_bid) / mid * 10_000,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "book_mid": mid,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
    }


def trade_flow_features(trades):
    if not trades:
        return {}

    buy_size = 0.0
    sell_size = 0.0
    total_size = 0.0

    # Coinbase trade objects include side/size/price.
    for t in trades:
        size = float(t.get("size", 0))
        side = str(t.get("side", "")).lower()
        total_size += size
        if side == "buy":
            buy_size += size
        elif side == "sell":
            sell_size += size

    imbalance = (
        (buy_size - sell_size) / total_size
        if total_size > 0 else np.nan
    )

    return {
        "trade_buy_size": buy_size,
        "trade_sell_size": sell_size,
        "trade_flow_imbalance": imbalance,
        "recent_trade_volume": total_size,
    }
