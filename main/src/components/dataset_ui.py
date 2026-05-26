"""Dataset upload, preview, and descriptive metrics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import dataset_stats
from utils.dataset_loader import REQUIRED_COLUMNS, load_dataset_from_bytes, load_dataset_from_path
from utils.file_storage import (
    get_default_dataset_path,
    load_active_dataset,
    log_dataset_import,
    save_active_dataset,
)


def _persist_dataset(df: pd.DataFrame, filename: str) -> None:
    st.session_state.dataset_df = df
    st.session_state.dataset_filename = filename
    save_active_dataset(df)
    missing = dataset_stats.missing_per_column(df)
    memory = dataset_stats.memory_usage_bytes(df)
    entry = log_dataset_import(
        filename=filename,
        row_count=len(df),
        column_count=len(df.columns),
        memory_bytes=memory,
        missing_values=missing.astype(int).to_dict(),
    )
    st.success(
        f"Loaded **{filename}** ({len(df):,} rows). "
        f"Saved locally (import #{entry['id']})."
    )


def render_dataset_section() -> None:
    st.header("Dataset Handling")
    st.caption(
        "Import Dogecoin daily OHLCV data (CSV or Excel). "
        f"Required columns: {', '.join(REQUIRED_COLUMNS)}. "
        "Extra columns (e.g. SNo, Symbol, Marketcap) are ignored."
    )

    default_path = get_default_dataset_path()
    col_load, col_upload = st.columns([1, 2])

    with col_load:
        if default_path.exists():
            if st.button("Load project Dogecoin dataset", use_container_width=True):
                try:
                    df = load_dataset_from_path(default_path)
                    _persist_dataset(df, default_path.name)
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        else:
            st.caption("Bundled `coin_Dogecoin.csv` not found in `main/data/`.")

    with col_upload:
        uploaded = st.file_uploader(
            "Or upload market data",
            type=["csv", "xlsx", "xls"],
            help="File must include Date, Open, High, Low, Close, Volume.",
        )

    if uploaded is not None:
        try:
            df = load_dataset_from_bytes(uploaded.getvalue(), uploaded.name)
            _persist_dataset(df, uploaded.name)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            return

    if "dataset_df" not in st.session_state:
        cached = load_active_dataset()
        if cached is not None:
            st.session_state.dataset_df = cached
            st.session_state.dataset_filename = "active_dataset.parquet (saved)"

    if "dataset_df" in st.session_state:
        _render_loaded_dataset(
            st.session_state.dataset_df,
            st.session_state.get("dataset_filename", "session"),
        )
    else:
        st.warning(
            "Load the project dataset or upload a CSV/Excel file to begin."
        )


def _render_loaded_dataset(df: pd.DataFrame, filename: str) -> None:
    st.subheader("Raw data")
    st.caption(f"Source: **{filename}**")

    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    st.subheader("Dataset metrics")
    missing = dataset_stats.missing_per_column(df)
    memory = dataset_stats.memory_usage_bytes(df)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total rows", f"{len(df):,}")
    m2.metric("Total columns", len(df.columns))
    m3.metric("Memory usage", f"{memory / 1024:.2f} KB")

    st.markdown("**Data types**")
    dtype_df = pd.DataFrame(
        {"column": df.dtypes.index.astype(str), "dtype": df.dtypes.astype(str).values}
    )
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("**Missing values (per column)**")
    missing_df = pd.DataFrame(
        {"column": missing.index, "missing_count": missing.values}
    )
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.markdown("**Statistical summary** (numerical features)")
    summary = dataset_stats.statistical_summary(df)
    if summary.empty:
        st.warning("No numerical columns available for summary statistics.")
    else:
        st.dataframe(summary, use_container_width=True)
