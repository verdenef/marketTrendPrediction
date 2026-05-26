"""Local file-based persistence for datasets and import logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LOGS_DIR = DATA_DIR / "logs"
PROCESSED_DIR = DATA_DIR / "processed"
ACTIVE_DATASET_PATH = DATA_DIR / "active_dataset.csv"
IMPORT_LOG_PATH = LOGS_DIR / "dataset_imports.json"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "processed_dataset.csv"
PREPROCESSING_LOG_PATH = LOGS_DIR / "preprocessing_runs.json"
TRAINING_LOG_PATH = LOGS_DIR / "training_history.json"
EVALUATION_LOG_PATH = LOGS_DIR / "evaluation_history.json"
DEFAULT_DATASET_PATH = DATA_DIR / "coin_Dogecoin.csv"


def ensure_storage_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_active_dataset(df: pd.DataFrame) -> Path:
    """Persist the current working dataset for reuse across sessions."""
    ensure_storage_dirs()
    df.to_csv(ACTIVE_DATASET_PATH, index=False)
    return ACTIVE_DATASET_PATH


def load_active_dataset() -> Optional[pd.DataFrame]:
    if not ACTIVE_DATASET_PATH.exists():
        return None
    from utils.dataset_loader import load_dataset_from_path

    return load_dataset_from_path(ACTIVE_DATASET_PATH)


def log_dataset_import(
    filename: str,
    row_count: int,
    column_count: int,
    memory_bytes: int,
    missing_values: dict[str, int],
) -> dict[str, Any]:
    """Append an import record to the local JSON log."""
    ensure_storage_dirs()
    records: list[dict[str, Any]] = []
    if IMPORT_LOG_PATH.exists():
        with IMPORT_LOG_PATH.open(encoding="utf-8") as f:
            records = json.load(f)

    entry = {
        "id": len(records) + 1,
        "filename": filename,
        "row_count": row_count,
        "column_count": column_count,
        "memory_bytes": memory_bytes,
        "missing_values": missing_values,
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    records.append(entry)
    with IMPORT_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return entry


def get_default_dataset_path() -> Path:
    return DEFAULT_DATASET_PATH


def save_processed_dataset(df: pd.DataFrame) -> Path:
    ensure_storage_dirs()
    df.to_csv(PROCESSED_DATASET_PATH, index=False)
    return PROCESSED_DATASET_PATH


def load_processed_dataset() -> Optional[pd.DataFrame]:
    if not PROCESSED_DATASET_PATH.exists():
        return None
    return pd.read_csv(PROCESSED_DATASET_PATH, parse_dates=["Date"])


def log_preprocessing_run(
    *,
    row_count: int,
    feature_count: int,
    train_rows: int,
    test_rows: int,
    train_ratio: float,
    scaling_method: str,
    rows_dropped: int,
    feature_columns: list[str],
) -> dict[str, Any]:
    ensure_storage_dirs()
    records: list[dict[str, Any]] = []
    if PREPROCESSING_LOG_PATH.exists():
        with PREPROCESSING_LOG_PATH.open(encoding="utf-8") as f:
            records = json.load(f)

    entry = {
        "id": len(records) + 1,
        "row_count": row_count,
        "feature_count": feature_count,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_ratio": train_ratio,
        "scaling_method": scaling_method,
        "rows_dropped": rows_dropped,
        "feature_columns": feature_columns,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    records.append(entry)
    with PREPROCESSING_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return entry


def log_training_run(
    *,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    train_rows: int,
    test_rows: int,
) -> dict[str, Any]:
    ensure_storage_dirs()
    records: list[dict[str, Any]] = []
    if TRAINING_LOG_PATH.exists():
        with TRAINING_LOG_PATH.open(encoding="utf-8") as f:
            records = json.load(f)

    entry = {
        "id": len(records) + 1,
        "model": model_name,
        "params": params,
        "metrics": metrics,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    records.append(entry)
    with TRAINING_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return entry


def log_evaluation_run(
    *,
    metrics_by_model: dict[str, dict[str, float]],
    training_history_ids: dict[str, int],
    preprocessing_meta: dict[str, Any],
) -> dict[str, Any]:
    """Append a comparative evaluation run into evaluation_history.json."""
    ensure_storage_dirs()

    records: list[dict[str, Any]] = []
    if EVALUATION_LOG_PATH.exists():
        with EVALUATION_LOG_PATH.open(encoding="utf-8") as f:
            records = json.load(f)

    entry = {
        "id": len(records) + 1,
        "metrics_by_model": metrics_by_model,
        "training_history_ids": training_history_ids,
        "preprocessing_meta": preprocessing_meta,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    records.append(entry)
    with EVALUATION_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return entry
