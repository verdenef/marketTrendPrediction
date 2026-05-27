"""KNN, SVM, and ANN model training UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from models.ann_model import parse_hidden_layers, train_ann
from models.knn_model import evaluate_knn, train_knn
from models.svm_model import evaluate_svm, train_svm
from utils.file_storage import log_training_run
from utils.model_cache import data_cache_key, set_cached_model
from utils.train_data import resolve_train_test_data

# Demo presets for one-click grading runs
KNN_PRESET = {"n_neighbors": 5, "metric": "euclidean"}
SVM_PRESET = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
ANN_PRESET = {
    "hidden_layers": [64, 32],
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
}

# Alternative preset set with clearly different defaults
KNN_ALT_PRESET = {"n_neighbors": 11, "metric": "manhattan"}
SVM_ALT_PRESET = {"kernel": "linear", "C": 0.5, "gamma": "scale"}
ANN_ALT_PRESET = {
    "hidden_layers": [128, 64, 16],
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.0005,
}


def _missing_preprocessing_warning() -> None:
    st.warning(
        "Preprocessed train/test data not found. Go to **Preprocessing** and run the "
        "full pipeline, or ensure `main/data/processed/processed_dataset.csv` and "
        "`main/data/logs/preprocessing_runs.json` exist."
    )


def _train_and_log_knn(X_train, X_test, y_train, y_test, data_key: str, **params) -> float:
    result = train_knn(
        X_train,
        y_train,
        n_neighbors=int(params["n_neighbors"]),
        metric=str(params["metric"]),
    )
    accuracy = evaluate_knn(result.model, X_test, y_test)
    set_cached_model("knn", result.params, data_key, result.model)
    st.session_state.knn_model = result.model
    st.session_state.knn_accuracy = accuracy
    log_training_run(
        model_name="knn",
        params=result.params,
        metrics={"accuracy": accuracy},
        train_rows=len(X_train),
        test_rows=len(X_test),
    )
    return accuracy


def _train_and_log_svm(X_train, X_test, y_train, y_test, data_key: str, **params) -> float:
    result = train_svm(
        X_train,
        y_train,
        kernel=str(params["kernel"]),
        C=float(params["C"]),
        gamma=params["gamma"],
    )
    accuracy = evaluate_svm(result.model, X_test, y_test)
    set_cached_model("svm", result.params, data_key, result.model)
    st.session_state.svm_model = result.model
    st.session_state.svm_accuracy = accuracy
    log_training_run(
        model_name="svm",
        params=result.params,
        metrics={"accuracy": accuracy},
        train_rows=len(X_train),
        test_rows=len(X_test),
    )
    return accuracy


def _train_and_log_ann(X_train, X_test, y_train, y_test, data_key: str, **params) -> float:
    result = train_ann(
        X_train,
        y_train,
        X_test,
        y_test,
        hidden_layers=params["hidden_layers"],
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
    )
    set_cached_model("ann", result.params, data_key, result.model)
    st.session_state.ann_model = result.model
    st.session_state.ann_accuracy = result.test_accuracy
    st.session_state.ann_history = result.history
    log_params = {k: v for k, v in result.params.items() if k != "loss_history"}
    log_params["loss_history_epochs"] = len(result.params.get("loss_history", []))
    log_training_run(
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
    return result.test_accuracy


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
    data_key = data_cache_key(X_train, feature_columns)

    st.success(
        f"Ready: **{len(X_train):,}** train / **{len(X_test):,}** test rows, "
        f"**{len(feature_columns)}** features."
    )

    st.info(
        "**Demo presets:** KNN K=5 (euclidean), SVM RBF (C=1), ANN layers 64,32 (20 epochs). "
        "Trained models are cached in memory for faster Evaluation and Live Inference."
    )

    preset_choice = st.selectbox(
        "Preset pack",
        options=["Demo preset (default)", "Alternative preset"],
        key="training_preset_pack",
    )
    use_alt_preset = preset_choice == "Alternative preset"
    knn_preset = KNN_ALT_PRESET if use_alt_preset else KNN_PRESET
    svm_preset = SVM_ALT_PRESET if use_alt_preset else SVM_PRESET
    ann_preset = ANN_ALT_PRESET if use_alt_preset else ANN_PRESET

    if st.button("Apply selected preset to form fields", type="secondary"):
        st.session_state.knn_k = int(knn_preset["n_neighbors"])
        st.session_state.knn_metric = str(knn_preset["metric"])
        st.session_state.svm_kernel = str(svm_preset["kernel"])
        st.session_state.svm_c = float(svm_preset["C"])
        st.session_state.svm_gamma = svm_preset["gamma"]
        st.session_state.ann_hidden = ",".join(str(n) for n in ann_preset["hidden_layers"])
        st.session_state.ann_epochs = int(ann_preset["epochs"])
        st.session_state.ann_batch = int(ann_preset["batch_size"])
        st.session_state.ann_lr = float(ann_preset["learning_rate"])
        st.rerun()

    preset_label = "alternative preset" if use_alt_preset else "demo preset"
    if st.button("Train all models (selected preset)", type="secondary"):
        with st.spinner(f"Training KNN, SVM, and ANN with {preset_label}..."):
            try:
                knn_acc = _train_and_log_knn(
                    X_train, X_test, y_train, y_test, data_key, **knn_preset
                )
                svm_acc = _train_and_log_svm(
                    X_train, X_test, y_train, y_test, data_key, **svm_preset
                )
                ann_acc = _train_and_log_ann(
                    X_train, X_test, y_train, y_test, data_key, **ann_preset
                )
            except ImportError as exc:
                st.error(f"ANN failed (TensorFlow): {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()
        st.success(
            f"All models trained. KNN={knn_acc:.4f}, SVM={svm_acc:.4f}, ANN={ann_acc:.4f}"
        )
        st.rerun()

    col_knn, col_svm = st.columns(2)

    with col_knn:
        st.subheader("K-Nearest Neighbors")
        n_neighbors = st.number_input(
            "n_neighbors (K)",
            min_value=1,
            max_value=50,
            value=KNN_PRESET["n_neighbors"],
            key="knn_k",
        )
        metric = st.selectbox(
            "Distance metric",
            options=["euclidean", "manhattan", "minkowski"],
            index=["euclidean", "manhattan", "minkowski"].index(KNN_PRESET["metric"]),
            key="knn_metric",
        )
        if st.button("Train KNN", type="primary", key="train_knn"):
            accuracy = _train_and_log_knn(
                X_train,
                X_test,
                y_train,
                y_test,
                data_key,
                n_neighbors=int(n_neighbors),
                metric=metric,
            )
            st.success(f"KNN trained. Test accuracy: **{accuracy:.4f}**.")

        if "knn_accuracy" in st.session_state:
            st.metric("KNN test accuracy", f"{st.session_state.knn_accuracy:.4f}")

    with col_svm:
        st.subheader("Support Vector Machine")
        kernel = st.selectbox(
            "Kernel",
            options=["linear", "rbf", "poly", "sigmoid"],
            index=["linear", "rbf", "poly", "sigmoid"].index(SVM_PRESET["kernel"]),
            key="svm_kernel",
        )
        C = st.number_input(
            "C (regularization)",
            min_value=0.01,
            max_value=100.0,
            value=float(SVM_PRESET["C"]),
            step=0.1,
            key="svm_c",
        )
        gamma_options = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
        gamma = st.selectbox(
            "gamma",
            options=gamma_options,
            index=gamma_options.index(SVM_PRESET["gamma"]),
            key="svm_gamma",
        )

        if st.button("Train SVM", type="primary", key="train_svm"):
            accuracy = _train_and_log_svm(
                X_train,
                X_test,
                y_train,
                y_test,
                data_key,
                kernel=kernel,
                C=float(C),
                gamma=gamma,
            )
            st.success(f"SVM trained. Test accuracy: **{accuracy:.4f}**.")

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
            value=ANN_PRESET["epochs"],
            key="ann_epochs",
        )
    with ann_col2:
        batch_size = st.number_input(
            "Batch size",
            min_value=8,
            max_value=512,
            value=ANN_PRESET["batch_size"],
            step=8,
            key="ann_batch",
        )
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0001,
            max_value=0.1,
            value=float(ANN_PRESET["learning_rate"]),
            format="%.4f",
            step=0.0001,
            key="ann_lr",
        )

    if st.button("Train ANN", type="primary", key="train_ann"):
        try:
            hidden_layers = parse_hidden_layers(hidden_layers_text)
            with st.spinner("Training neural network..."):
                accuracy = _train_and_log_ann(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    data_key,
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

        st.success(f"ANN trained. Test accuracy: **{accuracy:.4f}**.")

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
