"""Resolve train/test matrices from session or processed files."""

from __future__ import annotations

import json
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from utils.file_storage import PREPROCESSING_LOG_PATH, load_processed_dataset
from utils.preprocessing import TARGET_COLUMN

TrainTestBundle = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]


def _load_last_preprocessing_meta() -> Optional[dict]:
    if not PREPROCESSING_LOG_PATH.exists():
        return None
    with PREPROCESSING_LOG_PATH.open(encoding="utf-8") as f:
        records = json.load(f)
    return records[-1] if records else None


def splits_from_processed(
    processed: pd.DataFrame,
    train_rows: int,
    feature_columns: list[str],
) -> TrainTestBundle:
    if train_rows < 1 or train_rows >= len(processed):
        raise ValueError("Invalid train_rows in preprocessing log.")
    train_df = processed.iloc[:train_rows]
    test_df = processed.iloc[train_rows:]
    X_train = train_df[feature_columns].reset_index(drop=True)
    X_test = test_df[feature_columns].reset_index(drop=True)
    y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
    y_test = test_df[TARGET_COLUMN].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, feature_columns


def restore_splits_from_disk() -> Optional[TrainTestBundle]:
    processed = load_processed_dataset()
    meta = _load_last_preprocessing_meta()
    if processed is None or meta is None:
        return None
    feature_columns = meta.get("feature_columns")
    train_rows = meta.get("train_rows")
    if not feature_columns or train_rows is None:
        return None
    missing = [c for c in feature_columns if c not in processed.columns]
    if missing or TARGET_COLUMN not in processed.columns:
        return None
    return splits_from_processed(processed, int(train_rows), feature_columns)


def resolve_train_test_data() -> Optional[TrainTestBundle]:
    if all(
        key in st.session_state
        for key in ("X_train", "X_test", "y_train", "y_test", "feature_columns")
    ):
        return (
            st.session_state.X_train,
            st.session_state.X_test,
            st.session_state.y_train,
            st.session_state.y_test,
            st.session_state.feature_columns,
        )

    bundle = restore_splits_from_disk()
    if bundle is None:
        return None

    X_train, X_test, y_train, y_test, feature_columns = bundle
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.feature_columns = feature_columns
    st.session_state.processed_df = load_processed_dataset()
    return bundle
