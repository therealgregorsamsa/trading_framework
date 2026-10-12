import numpy as np
import pandas as pd

from hffx.config import FEATURE_COLUMNS
from hffx.features import (
    add_labels,
    add_returns,
    add_rolling_features,
    add_volatility,
    fill_missing_seconds,
)


def make_bars(n: int) -> pd.DataFrame:
    """One session of n regular 1-second bars with a random-walk close."""
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-5, n))
    return pd.DataFrame({
        "bar_ts": pd.date_range("2024-01-01 10:00:00", periods=n, freq="1s"),
        "symbol": "EURUSD",
        "session_id": 1,
        "n_quotes": rng.integers(1, 5, n),
        "open": close, "high": close, "low": close, "mean": close,
        "close": close,
        "spread": 1.4e-4,
    })


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    return add_rolling_features(add_volatility(add_returns(bars)))


def test_fill_missing_seconds_uses_past_price_only():
    bars = pd.DataFrame({
        "bar_ts": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:00:03"]),
        "symbol": ["EURUSD", "EURUSD"],
        "session_id": [1, 1],
        "n_quotes": [5, 2],
        "open": [1.10, 1.12], "high": [1.10, 1.12], "low": [1.10, 1.12],
        "mean": [1.10, 1.12], "close": [1.10, 1.12],
        "spread": [0.0001, 0.0002],
    })
    filled = fill_missing_seconds(bars)

    assert len(filled) == 4
    assert list(filled["n_quotes"]) == [5, 0, 0, 2]
    # missing seconds carry the LAST known price, never the next one (no look-ahead)
    assert list(filled["close"]) == [1.10, 1.10, 1.10, 1.12]
    assert list(filled["open"]) == [1.10, 1.10, 1.10, 1.12]
    assert list(filled["spread"]) == [0.0001, 0.0001, 0.0001, 0.0002]


def test_features_do_not_use_the_future():
    bars = make_bars(700)
    full = build_features(bars)
    truncated = build_features(bars.iloc[:500].copy())

    # features at time t must be identical whether or not the future exists
    pd.testing.assert_frame_equal(
        full.iloc[:500][FEATURE_COLUMNS],
        truncated[FEATURE_COLUMNS],
    )


def test_labels_by_hand_and_nan_at_session_end():
    bars = make_bars(180)
    # flat price, then a +10 bp jump at second 60 (threshold is 3 bp)
    bars["close"] = np.where(np.arange(180) < 60, 1.10, 1.10 * 1.001)
    labelled = add_labels(bars)

    assert (labelled["label"].iloc[:60] == 1).all()      # jump is 60 s ahead
    assert (labelled["label"].iloc[60:120] == 0).all()   # flat afterwards
    assert labelled["label"].iloc[120:].isna().all()     # no known future: NaN, not 0