"""Dataset summary metrics for the Streamlit dataset view."""

from __future__ import annotations

import pandas as pd

from utils.dataset_loader import NUMERIC_COLUMNS


def memory_usage_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True).sum())


def missing_per_column(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, min, max, std for numerical OHLCV columns."""
    numeric = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if not numeric:
        return pd.DataFrame()
    stats = df[numeric].describe().loc[["mean", "min", "max", "std"]]
    return stats.round(4)
