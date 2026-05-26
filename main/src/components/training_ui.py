"""KNN, SVM, and ANN model training UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from models.ann_model import parse_hidden_layers, train_ann
from models.knn_model import evaluate_knn, train_knn
from models.svm_model import evaluate_svm, train_svm
from utils.file_storage import log_training_run
from utils.train_data import resolve_train_test_data


def _missing_preprocessing_warning() -> None:
    st.warning(
        "Preprocessed train/test data not found. Go to **Preprocessing** and run the "
        "full pipeline, or ensure `main/data/processed/processed_dataset.csv` and "
        "`main/data/logs/preprocessing_runs.json` exist."
    )


def render_training_section() -> None:
    st.header("Model Training (KNN, SVM & ANN)")
    st.caption(
        "Train classifiers on the chronological train split from preprocessing. "
        "No random shuffling — models learn from past data and are checked on future rows."
    )

    bundle = resolve_train_test_data()
    if bundle is None:
        _missing_preprocessing_warning()
        return

    X_train, X_test, y_train, y_test, feature_columns = bundle
    st.success(
        f"Ready: **{len(X_train):,}** train / **{len(X_test):,}** test rows, "
        f"**{len(feature_columns)}** features."
    )

    col_knn, col_svm = st.columns(2)

    with col_knn:
        st.subheader("K-Nearest Neighbors")
        n_neighbors = st.number_input(
            "n_neighbors (K)",
            min_value=1,
            max_value=50,
            value=5,
            key="knn_k",
        )
        metric = st.selectbox(
            "Distance metric",
            options=["euclidean", "manhattan", "minkowski"],
            key="knn_metric",
        )
        if st.button("Train KNN", type="primary", key="train_knn"):
            result = train_knn(
                X_train,
                y_train,
                n_neighbors=int(n_neighbors),
                metric=metric,
            )
            accuracy = evaluate_knn(result.model, X_test, y_test)
            st.session_state.knn_model = result.model
            st.session_state.knn_accuracy = accuracy

            entry = log_training_run(
                model_name="knn",
                params=result.params,
                metrics={"accuracy": accuracy},
                train_rows=len(X_train),
                test_rows=len(X_test),
            )
            st.success(f"KNN trained. Test accuracy: **{accuracy:.4f}** (log #{entry['id']}).")

        if "knn_accuracy" in st.session_state:
            st.metric("KNN test accuracy", f"{st.session_state.knn_accuracy:.4f}")

    with col_svm:
        st.subheader("Support Vector Machine")
        kernel = st.selectbox(
            "Kernel",
            options=["linear", "rbf", "poly", "sigmoid"],
            key="svm_kernel",
        )
        C = st.number_input(
            "C (regularization)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.1,
            key="svm_c",
        )
        gamma_options = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
        gamma = st.selectbox("gamma", options=gamma_options, key="svm_gamma")

        if st.button("Train SVM", type="primary", key="train_svm"):
            result = train_svm(
                X_train,
                y_train,
                kernel=kernel,
                C=float(C),
                gamma=gamma,
            )
            accuracy = evaluate_svm(result.model, X_test, y_test)
            st.session_state.svm_model = result.model
            st.session_state.svm_accuracy = accuracy

            entry = log_training_run(
                model_name="svm",
                params=result.params,
                metrics={"accuracy": accuracy},
                train_rows=len(X_train),
                test_rows=len(X_test),
            )
            st.success(f"SVM trained. Test accuracy: **{accuracy:.4f}** (log #{entry['id']}).")

        if "svm_accuracy" in st.session_state:
            st.metric("SVM test accuracy", f"{st.session_state.svm_accuracy:.4f}")

    st.divider()
    st.subheader("Artificial Neural Network (TensorFlow/Keras)")
    ann_col1, ann_col2 = st.columns(2)
    with ann_col1:
        hidden_layers_text = st.text_input(
            "Hidden layers (comma-separated neurons)",
            value="64,32",
            help="Example: 64,32,16 builds three ReLU hidden layers.",
            key="ann_hidden",
        )
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=200,
            value=20,
            key="ann_epochs",
        )
    with ann_col2:
        batch_size = st.number_input(
            "Batch size",
            min_value=8,
            max_value=512,
            value=32,
            step=8,
            key="ann_batch",
        )
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%.4f",
            step=0.0001,
            key="ann_lr",
        )

    if st.button("Train ANN", type="primary", key="train_ann"):
        try:
            hidden_layers = parse_hidden_layers(hidden_layers_text)
            with st.spinner("Training neural network..."):
                result = train_ann(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    hidden_layers=hidden_layers,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate),
                )
        except ValueError as exc:
            st.error(str(exc))
            return
        except ImportError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"ANN training failed: {exc}")
            return

        st.session_state.ann_model = result.model
        st.session_state.ann_accuracy = result.test_accuracy
        st.session_state.ann_history = result.history

        log_params = {
            k: v for k, v in result.params.items() if k != "loss_history"
        }
        log_params["loss_history_epochs"] = len(result.params.get("loss_history", []))

        entry = log_training_run(
            model_name="ann",
            params=log_params,
            metrics={
                "accuracy": result.test_accuracy,
                "final_loss": float(result.history["loss"][-1])
                if result.history.get("loss")
                else 0.0,
            },
            train_rows=len(X_train),
            test_rows=len(X_test),
        )
        st.success(
            f"ANN trained. Test accuracy: **{result.test_accuracy:.4f}** "
            f"(log #{entry['id']})."
        )

    if "ann_accuracy" in st.session_state:
        st.metric("ANN test accuracy", f"{st.session_state.ann_accuracy:.4f}")

    if st.session_state.get("ann_history"):
        history = st.session_state.ann_history
        st.markdown("**Training loss by epoch**")
        loss_df = pd.DataFrame({"loss": history.get("loss", [])})
        st.line_chart(loss_df, use_container_width=True)
        if history.get("val_loss"):
            val_df = pd.DataFrame({"val_loss": history["val_loss"]})
            st.markdown("**Validation loss by epoch**")
            st.line_chart(val_df, use_container_width=True)
