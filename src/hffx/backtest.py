"""Cost-aware backtest of walk-forward predictions (all PnL in basis points)."""

import numpy as np
import pandas as pd

from hffx.validation import iter_rounds


def backtest_round(test: pd.DataFrame, predictions) -> pd.DataFrame:
    """Compute gross, cost and net PnL (in bps) of trading the model's predictions.

    The price is the mid, so one full spread per round trip is charged
    whenever a position is taken.
    """
    position = predictions
    gross_bps = position * test["future_return_60s"] * 1e4
    cost_bps = np.where(position != 0, test["spread_bps"], 0.0)
    return test.assign(
        position=position,
        gross_bps=gross_bps,
        cost_bps=cost_bps,
        net_bps=gross_bps - cost_bps,
    )


def positions_from_proba(model, X_train, X_test, top_share: float) -> np.ndarray:
    """Trade only the most confident predicted moves; the cutoff is computed on the train set."""
    classes = list(model.classes_)
    i_up, i_down = classes.index(1.0), classes.index(-1.0)

    def confidence(X):
        proba = model.predict_proba(X)
        up, down = proba[:, i_up], proba[:, i_down]
        return np.maximum(up, down), np.where(up > down, 1, -1)

    conf_train, _ = confidence(X_train)
    cutoff = np.quantile(conf_train, 1 - top_share)
    conf_test, direction = confidence(X_test)
    return np.where(conf_test > cutoff, direction, 0)


def backtest_results(
    df: pd.DataFrame,
    model,
    top_share: float | None = None,
    rounds: slice = slice(None),
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest returning per-second PnL (bps); optional probability-based rule."""
    results = []
    for r in list(iter_rounds(df, feature_columns))[rounds]:
        model.fit(r.X_train, r.y_train)
        if top_share is None:
            positions = model.predict(r.X_test)
        else:
            positions = positions_from_proba(model, r.X_train, r.X_test, top_share)
        results.append(backtest_round(r.test, positions))
    return pd.concat(results, ignore_index=True)


def performance_summary(results: pd.DataFrame) -> pd.Series:
    """Per-session Sharpe, maximum drawdown and trade statistics of the net PnL (bps)."""
    per_session = results.groupby("session_id")["net_bps"].sum()
    cumulative = results["net_bps"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    trades = results[results["position"] != 0]
    return pd.Series({
        "total_net_bps": results["net_bps"].sum(),
        "sharpe_per_session": per_session.mean() / per_session.std(),
        "max_drawdown_bps": drawdown.min(),
        "n_trades": len(trades),
        "hit_rate": (trades["gross_bps"] > 0).mean(),
    })