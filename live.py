import time
from pathlib import Path
import joblib
import numpy as np

from config import HORIZONS, MODEL_DIR, REFRESH_SECONDS
from data import (
    get_spot_klines,
    get_product_book,
    get_recent_trades,
    get_ticker,
    orderbook_features,
    trade_flow_features,
)
from features import make_features


def load_models():
    models = {}
    for h in HORIZONS:
        path = Path(MODEL_DIR) / f"btc_{h}m.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python train.py` first."
            )
        models[h] = joblib.load(path)
    return models


def main():
    models = load_models()

    print("BTC live predictor started. Ctrl+C to stop.")

    while True:
        try:
            candles = get_spot_klines(limit=300)
            feat = make_features(candles)
            latest = feat.iloc[-1]

            book = orderbook_features(get_product_book())
            flow = trade_flow_features(get_recent_trades())
            ticker = get_ticker()

            print("\033[2J\033[H", end="")
            print("=== BTC SHORT-HORIZON RESEARCH MODEL ===")
            print(f"Price:       ${float(ticker['price']):,.2f}")
            print(f"RSI(14):     {latest['rsi_14']:.2f}")
            print(f"Volume x20:  {latest['vol_ratio_20']:.2f}")
            print(f"ATR:         {latest['atr_pct']:.4%}")
            print(f"VWAP dist:   {latest['dist_vwap']:.4%}")

            if book:
                print(f"Book imb.:   {book['book_imbalance']:+.3f}")
                print(f"Spread:      {book['spread_bps']:.2f} bps")

            if flow:
                print(f"Trade flow:  {flow['trade_flow_imbalance']:+.3f}")

            print("\nMODEL OUTPUT")
            for h in HORIZONS:
                bundle = models[h]
                X = latest[bundle["features"]].to_frame().T
                if X.isna().any(axis=None):
                    print(f"{h:>2}m: insufficient features")
                    continue

                p = float(bundle["model"].predict_proba(X)[0, 1])
                signal = "UP" if p >= 0.5 else "DOWN"
                print(
                    f"{h:>2}m: {signal:<4} | "
                    f"UP {p:.1%} | DOWN {1-p:.1%}"
                )

            print("\nResearch only — validate before relying on any output.")
            time.sleep(REFRESH_SECONDS)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            print(f"\nLive data error: {exc}")
            time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
