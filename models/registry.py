"""
Model registry — saves and loads trained XGBoost models to disk.
Models live in trained_models/ which is gitignored.
"""
import os
import json
from datetime import datetime
from typing import Optional
import joblib
import structlog

log = structlog.get_logger()

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# In-memory cache: ticker → {model, label_encoder, feature_cols, meta}
_registry: dict[str, dict] = {}


def save(ticker: str, train_result: dict, backtest_result: dict) -> None:
    """Persist a trained model + metadata to disk."""
    if "error" in train_result:
        log.warning("skipping_save_due_to_error", ticker=ticker, error=train_result["error"])
        return

    base = os.path.join(MODEL_DIR, ticker)
    os.makedirs(base, exist_ok=True)

    joblib.dump(train_result["model"],         os.path.join(base, "model.joblib"))
    joblib.dump(train_result["label_encoder"], os.path.join(base, "label_encoder.joblib"))

    meta = {
        "ticker":            ticker,
        "trained_at":        datetime.utcnow().isoformat(),
        "feature_cols":      train_result["feature_cols"],
        "n_folds":           train_result["n_folds"],
        "n_samples":         train_result["n_samples"],
        "avg_accuracy":      train_result["avg_accuracy"],
        "avg_dir_accuracy":  train_result["avg_dir_accuracy"],
        "top_features":      train_result["top_features"],
        "label_distribution": train_result["label_distribution"],
        "backtest":          backtest_result,
        "folds":             train_result["folds"],
    }
    with open(os.path.join(base, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    _registry[ticker] = {
        "model":         train_result["model"],
        "label_encoder": train_result["label_encoder"],
        "feature_cols":  train_result["feature_cols"],
        "meta":          meta,
    }
    log.info("model_saved", ticker=ticker, dir=base)


def load(ticker: str) -> Optional[dict]:
    """Load a model from disk into the in-memory registry. Returns None if not found."""
    if ticker in _registry:
        return _registry[ticker]

    base = os.path.join(MODEL_DIR, ticker)
    model_path = os.path.join(base, "model.joblib")
    if not os.path.exists(model_path):
        return None

    try:
        model = joblib.load(model_path)
        le    = joblib.load(os.path.join(base, "label_encoder.joblib"))
        with open(os.path.join(base, "meta.json")) as f:
            meta = json.load(f)

        _registry[ticker] = {
            "model":         model,
            "label_encoder": le,
            "feature_cols":  meta["feature_cols"],
            "meta":          meta,
        }
        log.info("model_loaded", ticker=ticker)
        return _registry[ticker]
    except Exception as e:
        log.warning("model_load_failed", ticker=ticker, error=str(e))
        return None


def load_all(tickers: list[str]) -> None:
    """Load models for all tickers at startup."""
    for t in tickers:
        load(t)


def get(ticker: str) -> Optional[dict]:
    """Return in-memory model entry or try loading from disk."""
    return _registry.get(ticker) or load(ticker)


def list_models() -> list[dict]:
    """Return metadata for all saved models."""
    results = []
    if not os.path.exists(MODEL_DIR):
        return results
    for ticker in os.listdir(MODEL_DIR):
        meta_path = os.path.join(MODEL_DIR, ticker, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                results.append(json.load(f))
    return results
