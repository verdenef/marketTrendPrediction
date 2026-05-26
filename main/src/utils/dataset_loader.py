"""Load and validate Dogecoin OHLCV datasets from CSV or Excel."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
NUMERIC_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PathLike = Union[str, Path]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match column names case-insensitively to the expected schema."""
    mapping = {c.lower(): c for c in REQUIRED_COLUMNS}
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in mapping:
            renamed[col] = mapping[key]
    return df.rename(columns=renamed)


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of missing required column names."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[list(REQUIRED_COLUMNS)].sort_values("Date").reset_index(drop=True)


def load_dataset(uploaded_file: BinaryIO, filename: str) -> pd.DataFrame:
    """
    Parse an uploaded CSV or Excel file into a validated DataFrame.

    Extra columns (e.g. SNo, Name, Symbol, Marketcap) are ignored after normalization.
    """
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")

    df = _normalize_columns(df)
    missing = validate_columns(df)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}."
        )
    return _finalize_dataframe(df)


def load_dataset_from_path(path: PathLike) -> pd.DataFrame:
    """Load a dataset from a file on disk."""
    path = Path(path)
    with path.open("rb") as f:
        return load_dataset(f, path.name)


def load_dataset_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Convenience wrapper for Streamlit UploadedFile buffers."""
    return load_dataset(io.BytesIO(data), filename)
