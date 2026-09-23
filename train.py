from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance

from config import HORIZONS, MODEL_DIR, DATA_DIR, MIN_TRAIN_ROWS, HISTORY_CANDLES
from data import get_spot_klines
from features import make_features, add_targets, feature_columns


def train_one(df, horizon):
    cols = feature_columns(df)
    target = f"target_{horizon}m"
    future_ret = f"future_ret_{horizon}m"

    work = df[cols + [target, future_ret]].dropna()

    if len(work) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Only {len(work)} usable rows for {horizon}m; "
            f"need at least {MIN_TRAIN_ROWS}."
        )

    split = int(len(work) * 0.80)
    train = work.iloc[:split]
    test = work.iloc[split:]

    X_train, y_train = train[cols], train[target]
    X_test, y_test = test[cols], test[target]

    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    p_up = model.predict_proba(X_test)[:, 1]
    pred = (p_up >= 0.5).astype(int)

    metrics = {
        "horizon_minutes": horizon,
        "rows": len(work),
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, p_up),
        "test_up_rate": y_test.mean(),
        "mean_predicted_up_probability": p_up.mean(),
    }

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": cols, "horizon": horizon},
        model_dir / f"btc_{horizon}m.joblib",
    )

    imp = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=3,
        random_state=42,
        scoring="roc_auc",
    )
    importance = (
        pd.DataFrame({
            "feature": cols,
            "importance": imp.importances_mean,
        })
        .sort_values("importance", ascending=False)
        .head(25)
    )
    importance.to_csv(
        model_dir / f"importance_{horizon}m.csv",
        index=False,
    )

    return metrics


def main():
    print(f"Downloading {HISTORY_CANDLES:,} BTC 1-minute candles...")
    raw = get_spot_klines(limit=HISTORY_CANDLES)

    if len(raw) < 300:
        raise RuntimeError(
            f"Only {len(raw)} candles were received. "
            "The data provider did not return enough history."
        )

    print(f"Received {len(raw):,} candles.")
    feat = make_features(raw)
    dataset = add_targets(feat, HORIZONS)

    Path(DATA_DIR).mkdir(exist_ok=True)
    dataset.to_csv(Path(DATA_DIR) / "btc_features.csv")

    metrics = []
    for h in HORIZONS:
        print(f"\nTraining {h}-minute model...")
        result = train_one(dataset, h)
        metrics.append(result)
        print(json.dumps(result, indent=2))

    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path(MODEL_DIR, "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("\n======================================")
    print("TRAINING COMPLETE")
    print("Models saved in ./models/")
    print("Dataset saved in ./data/")
    print("======================================")


if __name__ == "__main__":
    main()
