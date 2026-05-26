"""Resolve dataset state shared across Streamlit pages."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from utils.file_storage import load_active_dataset


def resolve_raw_dataset() -> Optional[pd.DataFrame]:
    if "dataset_df" in st.session_state:
        return st.session_state.dataset_df
    df = load_active_dataset()
    if df is not None:
        st.session_state.dataset_df = df
        st.session_state.setdefault("dataset_filename", "active_dataset.csv")
    return df
