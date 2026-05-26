"""Comparative evaluation dashboard for KNN/SVM/ANN (file-based, no DB)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from models.ann_model import train_ann
from models.knn_model import train_knn
from models.svm_model import train_svm
from utils.file_storage import (
    PREPROCESSING_LOG_PATH,
    TRAINING_LOG_PATH,
    log_evaluation_run,
)
from utils.train_data import resolve_train_test_data


def _latest_train_run_for_model(model_name: str) -> dict[str, Any] | None:
    if not TRAINING_LOG_PATH.exists():
        return None
    with TRAINING_LOG_PATH.open(encoding="utf-8") as f:
        runs = json.load(f)

    for run in reversed(runs):
        if run.get("model") == model_name:
            return run
    return None


def _latest_preprocessing_meta() -> dict[str, Any] | None:
    if not PREPROCESSING_LOG_PATH.exists():
        return None
    with PREPROCESSING_LOG_PATH.open(encoding="utf-8") as f:
        records = json.load(f)
    return records[-1] if records else None


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
    }


def _plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[0, 1],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    st.pyplot(fig)


def render_evaluation_section() -> None:
    st.header("Evaluation Dashboard (KNN vs SVM vs ANN)")
    st.caption(
        "Compares Accuracy, Precision, Recall, and F1-score using the identical chronological test split. "
        "Confusion matrices are rendered for each trained model."
    )

    bundle = resolve_train_test_data()
    if bundle is None:
        st.warning("No preprocessed train/test split found. Run **Preprocessing** first.")
        return

    X_train, X_test, y_train, y_test, feature_columns = bundle
    meta = _latest_preprocessing_meta()

    st.success(
        f"Using test set: **{len(X_test):,}** rows · features: **{len(feature_columns)}**"
    )

    knn_run = _latest_train_run_for_model("knn")
    svm_run = _latest_train_run_for_model("svm")
    ann_run = _latest_train_run_for_model("ann")

    available = {
        "knn": knn_run is not None,
        "svm": svm_run is not None,
        "ann": ann_run is not None,
    }

    st.subheader("Model availability")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("KNN", "Ready" if available["knn"] else "Not trained")
    col_b.metric("SVM", "Ready" if available["svm"] else "Not trained")
    col_c.metric("ANN", "Ready" if available["ann"] else "Not trained")

    if not available["knn"]:
        st.warning("KNN not found in `main/data/logs/training_history.json`. Train KNN first to evaluate it.")
    if not available["svm"]:
        st.warning("SVM not found in `main/data/logs/training_history.json`. Train SVM first to evaluate it.")
    if not available["ann"]:
        st.warning("ANN not found in `main/data/logs/training_history.json`. Train ANN first to evaluate it.")

    if not any(available.values()):
        st.error("No trained model runs found in `main/data/logs/training_history.json`.")
        return

    metrics_by_model: dict[str, dict[str, float]] = {}
    y_pred_by_model: dict[str, np.ndarray] = {}
    train_history_ids: dict[str, int] = {}

    if st.button("Run comparative evaluation", type="primary"):
        with st.spinner("Evaluating models (train -> predict -> metrics)..."):
            # KNN
            if knn_run is not None:
                params = knn_run.get("params", {})
                n_neighbors = int(params.get("n_neighbors", 5))
                metric = str(params.get("metric", "euclidean"))
                model = train_knn(X_train, y_train, n_neighbors=n_neighbors, metric=metric).model
                y_pred = model.predict(X_test)
                y_pred_by_model["knn"] = y_pred
                metrics_by_model["knn"] = _compute_metrics(y_test, y_pred)
                train_history_ids["knn"] = int(knn_run["id"])

            # SVM
            if svm_run is not None:
                params = svm_run.get("params", {})
                kernel = str(params.get("kernel", "rbf"))
                C = float(params.get("C", 1.0))
                gamma = params.get("gamma", "scale")
                model = train_svm(X_train, y_train, kernel=kernel, C=C, gamma=gamma).model
                y_pred = model.predict(X_test)
                y_pred_by_model["svm"] = y_pred
                metrics_by_model["svm"] = _compute_metrics(y_test, y_pred)
                train_history_ids["svm"] = int(svm_run["id"])

            # ANN
            if ann_run is not None:
                params = ann_run.get("params", {})
                hidden_layers = params.get("hidden_layers", [64, 32])
                epochs = int(params.get("epochs", 20))
                batch_size = int(params.get("batch_size", 32))
                learning_rate = float(params.get("learning_rate", 0.001))

                try:
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
                except ImportError as exc:
                    st.error(f"ANN evaluation skipped: {exc}")
                else:
                    probs = result.model.predict(X_test.values, verbose=0).flatten()
                    y_pred = (probs >= 0.5).astype(int)
                    y_pred_by_model["ann"] = y_pred
                    metrics_by_model["ann"] = _compute_metrics(y_test, y_pred)
                    train_history_ids["ann"] = int(ann_run["id"])

        if metrics_by_model:
            preprocess_info = {
                "train_rows": int(meta.get("train_rows")) if meta else None,
                "test_rows": int(meta.get("test_rows")) if meta else None,
                "train_ratio": meta.get("train_ratio") if meta else None,
                "scaling_method": meta.get("scaling_method") if meta else None,
                "feature_columns": meta.get("feature_columns") if meta else feature_columns,
            }
            log_evaluation_run(
                metrics_by_model=metrics_by_model,
                training_history_ids=train_history_ids,
                preprocessing_meta=preprocess_info,
            )

            st.session_state.evaluation_metrics_by_model = metrics_by_model
            st.session_state.evaluation_y_pred_by_model = y_pred_by_model

        st.rerun()

    # If we already computed metrics earlier in this session, render them.
    if "evaluation_metrics_by_model" in st.session_state:
        metrics_by_model = st.session_state.evaluation_metrics_by_model
        y_pred_by_model = st.session_state.get("evaluation_y_pred_by_model", {})

    if metrics_by_model:
        st.subheader("Comparative metrics (side-by-side)")
        rows = []
        for model_name, m in metrics_by_model.items():
            label = {"knn": "KNN", "svm": "SVM", "ann": "ANN"}.get(model_name, model_name)
            rows.append(
                {
                    "Model": label,
                    "Accuracy": m.get("accuracy", np.nan),
                    "Precision": m.get("precision", np.nan),
                    "Recall": m.get("recall", np.nan),
                    "F1-score": m.get("f1", np.nan),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Confusion matrices")
        cols = st.columns(3)
        for idx, model_name in enumerate(["knn", "svm", "ann"]):
            if model_name not in y_pred_by_model:
                cols[idx].warning(f"{model_name.upper()} not evaluated.")
                continue
            with cols[idx]:
                plot_title = {"knn": "KNN", "svm": "SVM", "ann": "ANN"}.get(model_name, model_name)
                _plot_confusion_matrix(y_test, y_pred_by_model[model_name], f"{plot_title} Confusion Matrix")

