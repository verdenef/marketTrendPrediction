"""Preprocessing and feature engineering UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.file_storage import load_processed_dataset, log_preprocessing_run, save_processed_dataset
from utils.preprocessing import impute_missing, run_preprocessing_pipeline
from utils.session_data import resolve_raw_dataset


def _missing_dataset_warning() -> None:
    st.warning(
        "No dataset loaded. Go to **Dataset**, click **Load project Dogecoin dataset** "
        "or upload a file, then return here."
    )


def render_preprocessing_section() -> None:
    st.header("Preprocessing & Feature Engineering")
    st.caption(
        "Transform OHLCV data into ML-ready features with technical indicators, "
        "a binary target label, scaling, and a chronological train/test split."
    )

    raw_df = resolve_raw_dataset()
    if raw_df is None:
        _missing_dataset_warning()
        return

    st.success(f"Using dataset with **{len(raw_df):,}** rows.")

    with st.expander("Step 1 — Missing value imputation", expanded=False):
        st.markdown("Apply **forward-fill**, then **backward-fill** on OHLCV columns.")
        if st.button("Run imputation preview"):
            preview = impute_missing(raw_df)
            missing_after = int(preview.isna().sum().sum())
            st.session_state.imputed_preview = preview
            st.info(f"Imputation complete. Remaining missing cells: {missing_after}")

    col_a, col_b = st.columns(2)
    with col_a:
        sma_window = st.number_input("SMA window", min_value=2, max_value=60, value=14)
        rsi_window = st.number_input("RSI window", min_value=2, max_value=60, value=14)
    with col_b:
        scaling_method = st.selectbox(
            "Feature scaling",
            options=["standard", "minmax"],
            format_func=lambda x: "StandardScaler" if x == "standard" else "MinMaxScaler",
        )
        train_ratio_pct = st.slider(
            "Train split (chronological)",
            min_value=50,
            max_value=95,
            value=80,
            help="No random shuffling — earlier rows are training, later rows are test.",
        )

    st.markdown(
        "**Pipeline steps:** impute → SMA, RSI, MACD, lagged Close → "
        r"target ($Target_t = 1$ if $Close_{t+1} > Close_t$) → drop indicator warm-up NaNs → "
        "scale on train only → chronological split."
    )

    if st.button("Run full preprocessing pipeline", type="primary"):
        try:
            result = run_preprocessing_pipeline(
                raw_df,
                train_ratio=train_ratio_pct / 100,
                scaling_method=scaling_method,  # type: ignore[arg-type]
                sma_window=int(sma_window),
                rsi_window=int(rsi_window),
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.session_state.processed_df = result.processed
        st.session_state.X_train = result.X_train
        st.session_state.X_test = result.X_test
        st.session_state.y_train = result.y_train
        st.session_state.y_test = result.y_test
        st.session_state.feature_columns = result.feature_columns

        save_processed_dataset(result.processed)
        entry = log_preprocessing_run(
            row_count=len(result.processed),
            feature_count=len(result.feature_columns),
            train_rows=len(result.X_train),
            test_rows=len(result.X_test),
            train_ratio=result.train_ratio,
            scaling_method=result.scaling_method,
            rows_dropped=result.rows_dropped,
            feature_columns=result.feature_columns,
        )
        st.success(f"Preprocessing complete (run #{entry['id']}). Saved to `main/data/processed/`.")

    _render_results()


def _render_results() -> None:
    processed = st.session_state.get("processed_df")
    if processed is None:
        processed = load_processed_dataset()
        if processed is not None:
            st.session_state.processed_df = processed

    if processed is None:
        return

    st.subheader("Engineered features (sample)")
    st.dataframe(processed.head(20), use_container_width=True, hide_index=True)

    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")

    if X_train is not None and X_test is not None:
        st.subheader("Train / test shapes")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("X_train", f"{X_train.shape[0]} × {X_train.shape[1]}")
        c2.metric("X_test", f"{X_test.shape[0]} × {X_test.shape[1]}")
        if y_train is not None:
            c3.metric("y_train", len(y_train))
        if y_test is not None:
            c4.metric("y_test", len(y_test))

        st.markdown("**Feature columns**")
        st.write(", ".join(st.session_state.get("feature_columns", X_train.columns.tolist())))

        with st.expander("Preview scaled train features"):
            st.dataframe(X_train.head(10), use_container_width=True, hide_index=True)
