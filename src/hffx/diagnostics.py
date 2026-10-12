"""Diagnostics: leakage demo, per-class metrics, feature importance, IC, edge by regime."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from hffx.config import FEATURE_COLUMNS
from hffx.validation import iter_rounds


def inspect_round(df: pd.DataFrame, model, round_index: int = -1) -> None:
    """Fit a model on one walk-forward round and print its confusion matrix and per-class metrics."""
    r = list(iter_rounds(df))[round_index]
    model.fit(r.X_train, r.y_train)
    predictions = model.predict(r.X_test)
    print(confusion_matrix(r.y_test, predictions, labels=[-1, 0, 1]))
    print(classification_report(r.y_test, predictions, labels=[-1, 0, 1], zero_division=0))


def random_split_score(df: pd.DataFrame, model, test_size: float = 0.25) -> float:
    """Accuracy on a random row-level split: neighbouring seconds leak between train and test."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS], df["label"], test_size=test_size, shuffle=True, random_state=0
    )
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def feature_importance(df: pd.DataFrame, model, n_repeats: int = 5) -> pd.Series:
    """Average permutation importance (drop in neg log loss) across walk-forward rounds."""
    importances = []
    for r in iter_rounds(df):
        model.fit(r.X_train, r.y_train)
        result = permutation_importance(
            model, r.X_test, r.y_test,
            scoring="neg_log_loss", n_repeats=n_repeats, random_state=0,
        )
        importances.append(pd.Series(result.importances_mean, index=r.X_test.columns))
    return pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)


def session_ic(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str = "future_return_60s",
) -> pd.DataFrame:
    """Spearman IC of each feature vs the target, computed per session, then averaged."""
    per_session = pd.DataFrame(
        [group[feature_columns].corrwith(group[target], method="spearman")
         for _, group in df.groupby("session_id")]
    )
    return pd.DataFrame({
        "mean_ic": per_session.mean(),
        "t_stat": per_session.mean() / (per_session.std() / np.sqrt(len(per_session))),
        "share_positive_sessions": (per_session > 0).mean(),
    }).round(4).sort_values("mean_ic", key=abs, ascending=False)

