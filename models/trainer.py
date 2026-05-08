"""
XGBoost model trainer with walk-forward validation.

Walk-forward is the only honest way to backtest a predictive model on financial
time series. We never allow future data to touch the training window.

Prediction target:
  - Forward 5-bar return (25 min on 5-min bars)
  - BUY  if fwd_ret > +0.15%  (above expected transaction cost)
  - SELL if fwd_ret < -0.15%
  - HOLD otherwise

Why these numbers:
  - Alpaca is commission-free but bid-ask spread averages ~0.05-0.10%
  - 0.15% threshold gives a comfortable edge buffer
  - Forward 5-bar (25 min) gives enough time for the position to develop
    without being too long for a day-trading horizon
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import structlog

from features.engineering import compute_features, FEATURE_COLS

log = structlog.get_logger()

FORWARD_BARS  = 5       # how many bars ahead to predict
THRESHOLD_PCT = 0.0015  # 0.15% — minimum edge after transaction costs

# Conservative XGBoost params — tuned for financial time series
# max_depth=4 and min_child_weight=10 prevent overfitting on small datasets
XGB_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 4,
    learning_rate    = 0.04,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    min_child_weight = 10,
    gamma            = 0.2,
    reg_alpha        = 0.1,
    reg_lambda       = 1.5,
    tree_method      = "hist",
    eval_metric      = "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
)


def _make_labels(close: pd.Series) -> pd.Series:
    """Forward N-bar return classified into buy / sell / hold."""
    fwd_ret = close.pct_change(FORWARD_BARS).shift(-FORWARD_BARS)
    labels = pd.Series("hold", index=close.index, dtype=str)
    labels[fwd_ret >  THRESHOLD_PCT] = "buy"
    labels[fwd_ret < -THRESHOLD_PCT] = "sell"
    return labels


def _bars_per_day(df: pd.DataFrame) -> float:
    if hasattr(df.index, "normalize"):
        dates = df.index.normalize().unique()
        if len(dates) > 1:
            return len(df) / len(dates)
    return 78.0   # 6.5 hours × 12 bars/hour at 5-min timeframe


def walk_forward_train(df: pd.DataFrame) -> dict:
    """
    Train XGBoost with walk-forward cross-validation.

    Returns a dict with:
      model, label_encoder, feature_cols, folds, summary metrics
    """
    df = compute_features(df)
    if df is None or len(df) < 200:
        return {"error": "insufficient data — need at least 200 bars"}

    df = df.copy()
    df["label"] = _make_labels(df["close"])

    # Available features (subset of FEATURE_COLS that exist in this df)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Drop NaNs in features or label; drop last FORWARD_BARS rows (no valid label)
    df = df.dropna(subset=feat_cols + ["label"]).iloc[:-FORWARD_BARS]

    if len(df) < 100:
        return {"error": "too many NaN rows after feature computation"}

    le = LabelEncoder()
    df["y"] = le.fit_transform(df["label"])

    bpd         = _bars_per_day(df)
    train_bars  = int(252  * bpd)   # ~1 trading year
    test_bars   = int(21   * bpd)   # ~1 trading month
    min_train   = int(60   * bpd)   # minimum 60 days to start

    # Use shorter training window if we don't have a full year
    if len(df) < train_bars + test_bars:
        train_bars = max(int(len(df) * 0.8), min_train)
        test_bars  = len(df) - train_bars

    folds       = []
    accuracies  = []
    dir_accs    = []

    i = train_bars
    fold_num = 0
    while i + test_bars <= len(df):
        fold_num += 1
        tr = df.iloc[i - train_bars : i]
        te = df.iloc[i : i + test_bars]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            tr[feat_cols], tr["y"],
            eval_set=[(te[feat_cols], te["y"])],
            verbose=False,
        )

        preds = model.predict(te[feat_cols])
        acc   = float((preds == te["y"].values).mean())

        # Directional accuracy — only on bars the model called buy or sell
        buy_enc  = int(le.transform(["buy"])[0])  if "buy"  in le.classes_ else -1
        sell_enc = int(le.transform(["sell"])[0]) if "sell" in le.classes_ else -1
        dir_mask = (te["y"] == buy_enc) | (te["y"] == sell_enc)
        dir_acc  = float((preds[dir_mask] == te["y"].values[dir_mask]).mean()) if dir_mask.sum() > 0 else 0.0

        class_dist = te["label"].value_counts(normalize=True).to_dict()
        folds.append({
            "fold":          fold_num,
            "train_start":   str(tr.index[0]),
            "train_end":     str(tr.index[-1]),
            "test_start":    str(te.index[0]),
            "test_end":      str(te.index[-1]),
            "n_test":        len(te),
            "accuracy":      round(acc,     4),
            "dir_accuracy":  round(dir_acc, 4),
            "class_dist":    {k: round(v, 3) for k, v in class_dist.items()},
        })
        accuracies.append(acc)
        dir_accs.append(dir_acc)
        log.info("fold_complete", fold=fold_num, acc=round(acc, 3), dir_acc=round(dir_acc, 3))

        i += test_bars

    if not folds:
        return {"error": "could not form any walk-forward folds"}

    # Train final model on full dataset
    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(df[feat_cols], df["y"], verbose=False)

    # Feature importances
    importance = dict(zip(feat_cols, final_model.feature_importances_.tolist()))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "model":               final_model,
        "label_encoder":       le,
        "feature_cols":        feat_cols,
        "folds":               folds,
        "n_folds":             len(folds),
        "n_samples":           len(df),
        "avg_accuracy":        round(float(np.mean(accuracies)),  4),
        "avg_dir_accuracy":    round(float(np.mean(dir_accs)),    4),
        "top_features":        top_features,
        "label_distribution":  df["label"].value_counts(normalize=True).round(3).to_dict(),
    }


def predict_latest(model, le, feat_cols: list, df: pd.DataFrame) -> dict:
    """
    Run inference on the most recent bar.
    Returns action, confidence (0-100), and per-class probabilities.
    """
    df = compute_features(df)
    available = [c for c in feat_cols if c in df.columns]
    df = df.dropna(subset=available)

    if len(df) == 0:
        return {"action": "hold", "confidence": 0.0, "probabilities": {}}

    last   = df[available].iloc[[-1]]
    proba  = model.predict_proba(last)[0]
    pred_i = int(np.argmax(proba))
    action = str(le.classes_[pred_i])

    return {
        "action":        action,
        "confidence":    round(float(proba[pred_i]) * 100, 1),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)},
    }
