"""K-Nearest Neighbors classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class TrainResult:
    model: KNeighborsClassifier
    accuracy: float
    params: dict[str, Any]


def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_neighbors: int = 5,
    metric: str = "euclidean",
) -> TrainResult:
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    model.fit(X_train, y_train)
    return TrainResult(
        model=model,
        accuracy=0.0,
        params={"n_neighbors": n_neighbors, "metric": metric},
    )


def evaluate_knn(
    model: KNeighborsClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))
