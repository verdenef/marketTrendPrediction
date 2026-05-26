"""Artificial Neural Network (TensorFlow/Keras) for binary trading signals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

if TYPE_CHECKING:
    from tensorflow import keras as keras_types

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_TENSORFLOW_HELP = (
    "TensorFlow is required for ANN training. Install with Python 3.11 or 3.12: "
    "`pip install tensorflow>=2.15`. (TensorFlow does not yet support Python 3.14.)"
)


def _keras():
    try:
        from tensorflow import keras

        return keras
    except ImportError as exc:
        raise ImportError(_TENSORFLOW_HELP) from exc


@dataclass
class AnnTrainResult:
    model: Any
    params: dict[str, Any]
    history: dict[str, list[float]]
    test_accuracy: float


def parse_hidden_layers(spec: str) -> list[int]:
    """Parse comma-separated layer sizes, e.g. '64,32,16'."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("Provide at least one hidden layer size (e.g. 64,32).")
    layers = [int(p) for p in parts]
    if any(n < 1 for n in layers):
        raise ValueError("Each hidden layer must have at least 1 neuron.")
    return layers


def build_ann_model(
    input_dim: int,
    hidden_layers: list[int],
    learning_rate: float,
):
    keras = _keras()
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    hidden_layers: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> AnnTrainResult:
    model = build_ann_model(X_train.shape[1], hidden_layers, learning_rate)
    fit_history = model.fit(
        X_train.values,
        y_train.values,
        validation_data=(X_test.values, y_test.values),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    history = {k: [float(v) for v in vals] for k, vals in fit_history.history.items()}

    probabilities = model.predict(X_test.values, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)
    test_accuracy = float(accuracy_score(y_test, predictions))

    params = {
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "loss_history": history.get("loss", []),
    }
    return AnnTrainResult(
        model=model,
        params=params,
        history=history,
        test_accuracy=test_accuracy,
    )
