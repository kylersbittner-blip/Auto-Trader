"""
Walk-forward backtester — simulates trading the ML model's signals
with realistic costs (bid-ask spread, no commission on Alpaca).

Metrics returned:
  - total_return_pct
  - sharpe_ratio      (annualized, risk-free rate = 4.5%)
  - max_drawdown_pct
  - win_rate
  - profit_factor     (gross wins / gross losses)
  - n_trades
  - avg_trade_pct
"""
import numpy as np
import pandas as pd
import structlog

from features.engineering import compute_features, FEATURE_COLS
from models.trainer import _make_labels, FORWARD_BARS, XGB_PARAMS, _bars_per_day

log = structlog.get_logger()

SLIPPAGE_PCT  = 0.0005   # 0.05% estimated bid-ask (conservative for liquid stocks)
RISK_FREE_PCT = 0.045    # 4.5% annualised risk-free rate
BARS_PER_YEAR = 78 * 252 # 5-min bars in a trading year


def run_backtest(df: pd.DataFrame) -> dict:
    """
    Walk-forward backtest on the provided OHLCV DataFrame.

    Uses the same walk-forward methodology as the trainer to avoid lookahead.
    Each test period is traded using a model trained only on prior data.
    """
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    df = compute_features(df)
    if df is None or len(df) < 200:
        return {"error": "insufficient data for backtest"}

    df = df.copy()
    df["label"] = _make_labels(df["close"])
    feat_cols   = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=feat_cols + ["label"]).iloc[:-FORWARD_BARS]

    bpd        = _bars_per_day(df)
    train_bars = min(int(252 * bpd), int(len(df) * 0.8))
    test_bars  = max(int(21  * bpd), len(df) - train_bars)

    trade_returns = []
    equity_curve  = [1.0]
    peak          = 1.0

    i = train_bars
    while i + test_bars <= len(df):
        tr = df.iloc[i - train_bars : i]
        te = df.iloc[i : i + test_bars]

        le = LabelEncoder()
        y_tr = le.fit_transform(tr["label"])

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(tr[feat_cols], y_tr, verbose=False)

        proba     = model.predict_proba(te[feat_cols])
        pred_idx  = np.argmax(proba, axis=1)
        preds     = le.classes_[pred_idx]
        conf      = proba[np.arange(len(proba)), pred_idx]

        # Only trade when confidence exceeds 55% (filter low-conviction noise)
        for j, (action, confidence) in enumerate(zip(preds, conf)):
            if action == "hold" or confidence < 0.55:
                continue
            if i + j + FORWARD_BARS >= len(df):
                break

            entry_bar = df.iloc[i + j]
            exit_bar  = df.iloc[i + j + FORWARD_BARS]

            entry_price = entry_bar["close"] * (1 + SLIPPAGE_PCT)
            exit_price  = exit_bar["close"]  * (1 - SLIPPAGE_PCT)

            if action == "buy":
                raw_ret = (exit_price - entry_price) / entry_price
            else:  # sell / short
                raw_ret = (entry_price - exit_price) / entry_price

            trade_returns.append(raw_ret)
            equity_curve.append(equity_curve[-1] * (1 + raw_ret))
            peak = max(peak, equity_curve[-1])

        i += test_bars

    if not trade_returns:
        return {"error": "no trades generated — confidence threshold too high or insufficient signals"}

    returns_arr   = np.array(trade_returns)
    equity_arr    = np.array(equity_curve)

    total_return  = float(equity_arr[-1] - 1)
    wins          = returns_arr[returns_arr > 0]
    losses        = returns_arr[returns_arr < 0]
    win_rate      = float(len(wins) / len(returns_arr))
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    # Annualised Sharpe
    avg_ret       = float(returns_arr.mean())
    std_ret       = float(returns_arr.std())
    rf_per_trade  = RISK_FREE_PCT / BARS_PER_YEAR
    sharpe        = ((avg_ret - rf_per_trade) / std_ret * np.sqrt(BARS_PER_YEAR)) if std_ret > 0 else 0.0

    # Max drawdown
    running_max   = np.maximum.accumulate(equity_arr)
    drawdowns     = (equity_arr - running_max) / running_max
    max_drawdown  = float(drawdowns.min())

    return {
        "total_return_pct":  round(total_return  * 100, 2),
        "sharpe_ratio":      round(float(sharpe),        2),
        "max_drawdown_pct":  round(max_drawdown   * 100, 2),
        "win_rate_pct":      round(win_rate        * 100, 1),
        "profit_factor":     round(profit_factor,         2),
        "n_trades":          len(trade_returns),
        "avg_trade_pct":     round(float(avg_ret  * 100), 3),
        "equity_curve":      [round(v, 4) for v in equity_arr.tolist()[::10]],  # downsample
    }
