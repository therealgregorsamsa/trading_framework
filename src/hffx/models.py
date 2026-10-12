"""Model factories and walk-forward accuracy evaluation."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

from hffx.validation import iter_rounds


def baseline_model() -> DummyClassifier:
    """Always predicts the most frequent class of the training set."""
    return DummyClassifier(strategy="most_frequent")


def logistic_model() -> Pipeline:
    """Standardised logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def boosting_model() -> HistGradientBoostingClassifier:
    """Gradient boosting with shallow trees."""
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=0
    )


def evaluate_model(
    df: pd.DataFrame,
    model,
    feature_columns: list[str] | None = None,
) -> list[float]:
    """Fit a model on every walk-forward round and return one test accuracy per round."""
    scores = []
    for r in iter_rounds(df, feature_columns):
        model.fit(r.X_train, r.y_train)
        scores.append(accuracy_score(r.y_test, model.predict(r.X_test)))
    return scores