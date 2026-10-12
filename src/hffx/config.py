"""Project-wide constants."""

from pathlib import Path

MIN_TRAIN_SESSIONS = 3
FEATURES_PATH = Path("data/features/features_3bp.parquet")

FEATURE_COLUMNS = [
    "return_1s", "return_10s", "return_60s", "return_300s",
    "volatility_60s", "volatility_300s", "zscore_300s",
    "activity_60s", "n_quotes", "spread_bps", "hour",
]
