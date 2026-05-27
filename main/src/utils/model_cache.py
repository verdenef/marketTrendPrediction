"""In-memory model cache (Streamlit session) to avoid redundant retraining."""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd
import streamlit as st

from models.ann_model import train_ann
from models.knn_model import train_knn
from models.svm_model import train_svm


def data_cache_key(X_train: pd.DataFrame, feature_columns: list[str]) -> str:
    """Fingerprint the current preprocessed train matrix."""
    return f"{len(X_train)}_{len(feature_columns)}_{hash(tuple(feature_columns))}"


def _params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def invalidate_model_cache() -> None:
    st.session_state.model_cache = {}


def _bucket() -> dict[str, dict[str, Any]]:
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}
    return st.session_state.model_cache


def get_cached_model(
    model_name: str,
    params: dict[str, Any],
    data_key: str,
) -> Optional[Any]:
    entry = _bucket().get(model_name)
    if (
        entry
        and entry.get("params_key") == _params_key(params)
        and entry.get("data_key") == data_key
    ):
        return entry.get("model")
    return None


def set_cached_model(
    model_name: str,
    params: dict[str, Any],
    data_key: str,
    model: Any,
) -> None:
    _bucket()[model_name] = {
        "params_key": _params_key(params),
        "data_key": data_key,
        "model": model,
    }


def get_or_train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_neighbors: int,
    metric: str,
    data_key: str,
) -> Any:
    params = {"n_neighbors": n_neighbors, "metric": metric}
    cached = get_cached_model("knn", params, data_key)
    if cached is not None:
        return cached
    model = train_knn(
        X_train, y_train, n_neighbors=n_neighbors, metric=metric
    ).model
    set_cached_model("knn", params, data_key, model)
    return model


def get_or_train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    kernel: str,
    C: float,
    gamma: Any,
    data_key: str,
) -> Any:
    params = {"kernel": kernel, "C": C, "gamma": gamma}
    cached = get_cached_model("svm", params, data_key)
    if cached is not None:
        return cached
    model = train_svm(
        X_train, y_train, kernel=kernel, C=C, gamma=gamma
    ).model
    set_cached_model("svm", params, data_key, model)
    return model


def get_or_train_ann(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    hidden_layers: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    data_key: str,
) -> Any:
    params = {
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    cached = get_cached_model("ann", params, data_key)
    if cached is not None:
        return cached
    result = train_ann(
        X_train,
        y_train,
        X_test,
        y_test,
        hidden_layers=hidden_layers,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    set_cached_model("ann", params, data_key, result.model)
    return result.model
