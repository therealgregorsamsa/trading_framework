"""Load the model-ready dataset."""

from pathlib import Path

import pandas as pd

from hffx.config import FEATURE_COLUMNS, FEATURES_PATH


def load_dataset(
    symbol: str,
    path: Path = FEATURES_PATH,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one symbol's features, dropping rows with an incomplete feature set or no label."""
    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    df = pd.read_parquet(path)
    df = df[df["symbol"] == symbol]
    return df.dropna(subset=columns + ["label"]).reset_index(drop=True)
