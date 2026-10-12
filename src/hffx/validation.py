"""Walk-forward validation utilities."""
from hffx.config import FEATURE_COLUMNS, MIN_TRAIN_SESSIONS
import pandas as pd
from typing import Iterator, NamedTuple

def walk_forward_splits(sessions: list[int]) -> list[tuple[list[int], int]]:
    """Return (train sessions, test session) pairs for an expanding walk-forward.

    Round k trains on the first k sessions and tests on the next one, so the
    model never sees the session it predicts.

    Args:
        sessions: session ids in chronological order.

    Returns:
        A list of (train_sessions, test_session) pairs.
    """
    splits = []
    for k in range(MIN_TRAIN_SESSIONS, len(sessions)):
        splits.append((sessions[:k], sessions[k]))
    return splits

def split_round(df: pd.DataFrame, train_sessions: list, test_session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test rows for one walk-forward round."""
    train = df[df["session_id"].isin(train_sessions)]
    test = df[df["session_id"] == test_session]
    return train, test


def make_xy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Separate features (X) from the target label (y) for train and test.

    Args:
        train: rows of the training sessions.
        test: rows of the test session.
        feature_columns: columns used as features. Defaults to FEATURE_COLUMNS.

    Returns:
        X_train, y_train, X_test, y_test.
    """
    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    return train[columns], train["label"], test[columns], test["label"]


class Round(NamedTuple):
    """One walk-forward round: the raw rows and the matching X/y matrices."""
    test_session: int
    train: pd.DataFrame
    test: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def iter_rounds(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> Iterator[Round]:
    """Yield every expanding walk-forward round of a dataset, in chronological order.

    Args:
        df: dataset with a `session_id` column and a `label` column.
        feature_columns: columns used as features. Defaults to FEATURE_COLUMNS.

    Yields:
        One Round per (train sessions, test session) pair.
    """
    sessions = sorted(df["session_id"].unique())
    for train_sessions, test_session in walk_forward_splits(sessions):
        train, test = split_round(df, train_sessions, test_session)
        X_train, y_train, X_test, y_test = make_xy(train, test, feature_columns)
        yield Round(test_session, train, test, X_train, y_train, X_test, y_test)
