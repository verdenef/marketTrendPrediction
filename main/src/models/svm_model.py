"""Support Vector Machine classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


@dataclass
class TrainResult:
    model: SVC
    accuracy: float
    params: dict[str, Any]


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: Optional[str | float] = "scale",
) -> TrainResult:
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return TrainResult(
        model=model,
        accuracy=0.0,
        params={"kernel": kernel, "C": C, "gamma": gamma},
    )


def evaluate_svm(
    model: SVC,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    predictions = model.predict(X_test)
    return float(accuracy_score(y_test, predictions))
