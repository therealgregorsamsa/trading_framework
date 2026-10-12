from hffx.validation import iter_rounds, walk_forward_splits, split_round, make_xy
import pandas as pd

def test_walk_forward_splits_expected_rounds():
    sessions = [10, 11, 12, 13, 14]
    assert walk_forward_splits(sessions) == [
        ([10, 11, 12], 13),
        ([10, 11, 12, 13], 14),
    ]

def test_split_round_separates_sessions():
    df = pd.DataFrame({
        "session_id": [1, 1, 2, 2, 2, 3, 3],
        "value":      [10, 11, 20, 21, 22, 30, 31],
    })
    train, test = split_round(df, train_sessions=[1, 2], test_session=3)

    assert len(train) == 5
    assert len(test) == 2
    assert list(test["value"]) == [30, 31]
    assert set(train["session_id"]).isdisjoint(set(test["session_id"]))

def test_train_sessions_always_precede_test_session():
    sessions = [10, 11, 12, 13, 14, 15, 16]
    for train_sessions, test_session in walk_forward_splits(sessions):
        assert max(train_sessions) < test_session
        assert test_session not in train_sessions


def test_make_xy_excludes_identifiers_and_label():
    df = pd.DataFrame({
        "session_id": [1, 1, 2],
        "spread_bps": [1.4, 1.5, 1.4],
        "hour": [9, 10, 11],
        "label": [1, 0, -1],
    })
    train, test = df.iloc[:2], df.iloc[2:]
    X_train, y_train, X_test, y_test = make_xy(
        train, test, feature_columns=["spread_bps", "hour"]
    )

    assert list(X_train.columns) == ["spread_bps", "hour"]
    assert "session_id" not in X_train.columns
    assert "label" not in X_train.columns
    assert list(y_train) == [1, 0]
    assert list(y_test) == [-1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_make_xy_uses_default_features():
    from hffx.config import FEATURE_COLUMNS

    df = pd.DataFrame({column: [0.0, 1.0] for column in FEATURE_COLUMNS})
    df["label"] = [1, -1]
    X_train, _, _, _ = make_xy(df.iloc[:1], df.iloc[1:])

    assert list(X_train.columns) == FEATURE_COLUMNS

def make_dataset(n_sessions: int) -> pd.DataFrame:
    """Small dataset: 2 rows per session, one feature and a label."""
    return pd.DataFrame({
        "session_id": [s for s in range(1, n_sessions + 1) for _ in range(2)],
        "spread_bps": [1.4] * (2 * n_sessions),
        "label": [1, -1] * n_sessions,
    })


def test_iter_rounds_count_and_order():
    rounds = list(iter_rounds(make_dataset(6), feature_columns=["spread_bps"]))

    assert [r.test_session for r in rounds] == [4, 5, 6]


def test_iter_rounds_never_leaks_test_into_train():
    for r in iter_rounds(make_dataset(6), feature_columns=["spread_bps"]):
        assert r.train["session_id"].max() < r.test_session
        assert set(r.test["session_id"]) == {r.test_session}
        assert len(r.X_train) == len(r.y_train)
        assert len(r.X_test) == len(r.y_test)
