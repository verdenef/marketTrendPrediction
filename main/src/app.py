"""
Dogecoin Market Trend Prediction BI App — Streamlit entry point.

IS 108 Final Project · Caraga State University · SY 2025-2026
"""

from __future__ import annotations

import streamlit as st

from components.dataset_ui import render_dataset_section
from components.preprocessing_ui import render_preprocessing_section
from components.evaluation_ui import render_evaluation_section
from components.training_ui import render_training_section
from components.inference_ui import render_inference_section

st.set_page_config(
    page_title="DOGE BI — Market Trend Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Dataset": render_dataset_section,
    "Preprocessing": render_preprocessing_section,
    "Model Training": render_training_section,
    "Evaluation": render_evaluation_section,
    "Live Inference": render_inference_section,
}


def main() -> None:
    st.title("Dogecoin Market Trend Prediction")
    st.markdown(
        "Business Intelligence app for **trading signal classification** "
        "(Price UP / Buy vs Price DOWN / Sell) using historical DOGE market data."
    )

    with st.sidebar:
        st.header("Navigation")
        selection = st.radio(
            "Section",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Grader demo (5 steps)**")
        st.markdown(
            "1. Dataset → load project data\n"
            "2. Preprocessing → run pipeline\n"
            "3. Model Training → **Train all models**\n"
            "4. Evaluation → run comparison\n"
            "5. Live Inference → run + export report"
        )
        st.caption("Use `.venv312` (Python 3.12) for ANN/TensorFlow.")

    PAGES[selection]()


if __name__ == "__main__":
    main()
