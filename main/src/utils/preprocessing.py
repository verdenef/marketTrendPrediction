"""Preprocessing pipeline: imputation, features, target, scaling, chronological split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.technical_indicators import add_all_indicators

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
TARGET_COLUMN = "Target"
ScalingMethod = Literal["standard", "minmax"]


@dataclass
class PreprocessResult:
    processed: pd.DataFrame
    feature_columns: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_ratio: float
    scaling_method: ScalingMethod
    rows_dropped: int


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill along the time-ordered series."""
    out = df.sort_values("Date").copy()
    out[list(OHLCV_COLUMNS)] = out[list(OHLCV_COLUMNS)].ffill().bfill()
    return out


def add_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """Target_t = 1 if Close_{t+1} > Close_t else 0."""
    out = df.copy()
    out[TARGET_COLUMN] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    return out


def _indicator_feature_names(sma_window: int, rsi_window: int) -> list[str]:
    return [
        f"SMA_{sma_window}",
        f"RSI_{rsi_window}",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "Close_lag1",
    ]


def build_feature_columns(sma_window: int, rsi_window: int) -> list[str]:
    return list(OHLCV_COLUMNS) + _indicator_feature_names(sma_window, rsi_window)


def drop_indicator_warmup_rows(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.dropna(subset=feature_columns + [TARGET_COLUMN]).reset_index(drop=True)
    return cleaned, before - len(cleaned)


def run_feature_engineering(
    df: pd.DataFrame,
    sma_window: int = 14,
    rsi_window: int = 14,
) -> tuple[pd.DataFrame, list[str], int]:
    out = impute_missing(df)
    out = add_all_indicators(out, sma_window=sma_window, rsi_window=rsi_window)
    out = add_target_label(out)
    feature_columns = build_feature_columns(sma_window, rsi_window)
    out, dropped = drop_indicator_warmup_rows(out, feature_columns)
    return out, feature_columns, dropped


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    method: ScalingMethod,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_ratio <= 0.95:
        raise ValueError("Train ratio must be between 0.5 and 0.95.")
    split_idx = int(len(df) * train_ratio)
    if split_idx < 1 or split_idx >= len(df):
        raise ValueError("Train ratio yields an empty train or test set.")
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def assemble_processed_frame(
    source: pd.DataFrame,
    X_scaled: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": source["Date"].values})
    for col in feature_columns:
        frame[col] = X_scaled[col].values
    frame[TARGET_COLUMN] = source[TARGET_COLUMN].values
    return frame


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    scaling_method: ScalingMethod = "standard",
    sma_window: int = 14,
    rsi_window: int = 14,
) -> PreprocessResult:
    engineered, feature_columns, dropped = run_feature_engineering(
        df, sma_window=sma_window, rsi_window=rsi_window
    )
    train_raw, test_raw = chronological_split(engineered, train_ratio)
    X_train_scaled, X_test_scaled = scale_features(
        train_raw, test_raw, feature_columns, scaling_method
    )

    y_train = train_raw[TARGET_COLUMN].reset_index(drop=True)
    y_test = test_raw[TARGET_COLUMN].reset_index(drop=True)
    X_train = X_train_scaled.reset_index(drop=True)
    X_test = X_test_scaled.reset_index(drop=True)

    train_processed = assemble_processed_frame(
        train_raw.reset_index(drop=True), X_train, feature_columns
    )
    test_processed = assemble_processed_frame(
        test_raw.reset_index(drop=True), X_test, feature_columns
    )
    processed = pd.concat([train_processed, test_processed], ignore_index=True)

    return PreprocessResult(
        processed=processed,
        feature_columns=feature_columns,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_ratio=train_ratio,
        scaling_method=scaling_method,
        rows_dropped=dropped,
    )
