"""Feature layer: from 1-second bars to model-ready features and labels."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/features/features_3bp.parquet")
BARS_PATH = Path("data/bars/bars.parquet")

FILLED_FROM_CLOSE = ["open", "high", "low", "mean"]
RETURN_HORIZONS = [1, 10, 60, 300]  # in seconds
FEATURE_WINDOW = 300  # 5 minutes
LABEL_HORIZON = 60          # seconds ahead
LABEL_THRESHOLD_BPS = 3.0   # about the average spread



def fill_missing_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Add a bar for every second of one session that had no quote.

    A missing second gets the last known price (forward fill, past only:
    interpolating with the next bar would be look-ahead) and n_quotes = 0,
    which also flags it as filled.

    Args:
        df: bars of a single symbol and session, with a 'bar_ts' column.

    Returns:
        One bar per second between the first and last bar of the session.
    """
    full_index = pd.date_range(start=df["bar_ts"].min(), end=df["bar_ts"].max(), freq="1s")
    df = df.set_index("bar_ts").reindex(full_index)
    df.index.name = "bar_ts"
    df = df.assign(
        symbol=df["symbol"].ffill(),
        session_id=df["session_id"].ffill().astype("int64"),
        n_quotes=df["n_quotes"].fillna(0).astype("int64"),
        close=df["close"].ffill(),
        spread=df["spread"].ffill(),
    )
    for col in FILLED_FROM_CLOSE:
        df[col] = df[col].fillna(df["close"])
    return df.reset_index()


def fill_all_sessions(bars: pd.DataFrame) -> pd.DataFrame:
    """Apply fill_missing_seconds to each symbol and session separately.

    Filling session by session guarantees that market breaks
    (daily pause, weekends, gaps between files) are never filled.

    Args:
        bars: 1-second bars of all symbols and sessions.

    Returns:
        The bars with every missing second added.
    """
    filled = [fill_missing_seconds(g) for _, g in bars.groupby(["symbol", "session_id"])]
    return pd.concat(filled, ignore_index=True)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add past returns of the close over several horizons.

    Returns are computed within each symbol and session, so that no window
    crosses a market break. The first seconds of each session get NaN.

    Args:
        df: 1-second bars on a regular grid (after fill_all_sessions).

    Returns:
        The bars with one 'return_<h>s' column per horizon.
    """
    groups = df.groupby(["symbol", "session_id"])["close"]
    for h in RETURN_HORIZONS:
        previous_close = groups.shift(h)
        df = df.assign(**{f"return_{h}s": np.log(df["close"] / previous_close)})
    return df

def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add the realized volatility of 1-second log returns over the last minute.

    The rolling window stays within each symbol and session, so it never
    spans a market break. The first 60 seconds of each session get NaN.

    Args:
        df: bars with a 'return_1s' column (from add_returns).

    Returns:
        The bars with a 'volatility_60s' column.
    """
    groups = df.groupby(["symbol", "session_id"])["return_1s"]
    df = df.assign(
        volatility_60s=groups.transform(lambda s: s.rolling(60).std())
    )
    return df

FEATURE_WINDOW = 300  # 5 minutes


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features computed within each symbol and session.

    - volatility_300s: realized volatility of 1-second returns over 5 minutes
    - zscore_300s: distance of the close from its 5-minute mean, in standard
      deviations (mean reversion signal); NaN when the price was flat
    - activity_60s: number of quotes over the last minute
    - spread_bps: current spread in basis points of the price
    - hour: hour of the day (UTC), since behaviour changes across sessions

    Args:
        df: bars with return_1s (from add_returns).

    Returns:
        The bars with the new feature columns.
    """
    groups = df.groupby(["symbol", "session_id"])
    w = FEATURE_WINDOW
    close_mean = groups["close"].transform(lambda s: s.rolling(w).mean())
    close_std = groups["close"].transform(lambda s: s.rolling(w).std())
    return df.assign(
        volatility_300s=groups["return_1s"].transform(lambda s: s.rolling(w).std()),
        zscore_300s=((df["close"] - close_mean) / close_std).replace([np.inf, -np.inf], np.nan),
        activity_60s=groups["n_quotes"].transform(lambda s: s.rolling(60).sum()),
        spread_bps=df["spread"] / df["close"] * 1e4,
        hour=df["bar_ts"].dt.hour,
    )

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add the future return and the buy / sell / neutral label.

    The label looks LABEL_HORIZON seconds ahead, within each symbol and
    session: the last seconds of a session have no known future and keep
    a NaN label instead of being silently marked neutral.

    Label: 1 (buy) if the future log return exceeds the threshold,
    -1 (sell) if it is below minus the threshold, 0 (neutral) otherwise.
    The threshold is about one spread: smaller moves would not cover costs.

    Args:
        df: bars on a regular grid, with a 'close' column.

    Returns:
        The bars with 'future_return_60s' and 'label' columns.
    """
    groups = df.groupby(["symbol", "session_id"])["close"]
    future_close = groups.shift(-LABEL_HORIZON)
    future_return = np.log(future_close / df["close"])
    threshold = LABEL_THRESHOLD_BPS / 1e4
    label = np.select(
        [future_return.isna(), future_return > threshold, future_return < -threshold],
        [np.nan, 1, -1],
        default=0,
    )
    return df.assign(future_return_60s=future_return, label=label)



if __name__ == "__main__":
    bars = pd.read_parquet(BARS_PATH)
    df = fill_all_sessions(bars)
    df = add_returns(df)
    df = add_volatility(df)
    df = add_rolling_features(df)
    df = add_labels(df)

    new_cols = ["volatility_300s", "zscore_300s", "activity_60s", "spread_bps", "hour"]
    print(df.groupby("symbol")["label"].value_counts(dropna=False, normalize=True))
    print(df.groupby("symbol")["label"].apply(lambda s: s.isna().sum()))
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"{len(df)} rows written to {FEATURES_PATH}")
