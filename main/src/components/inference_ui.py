"""Live inference (BUY/SELL) and academic report export (file-based only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils.model_cache import (
    data_cache_key,
    get_or_train_ann,
    get_or_train_knn,
    get_or_train_svm,
)
from utils.file_storage import (
    EVALUATION_LOG_PATH,
    PREPROCESSING_LOG_PATH,
    TRAINING_LOG_PATH,
    load_processed_dataset,
)
from utils.train_data import resolve_train_test_data

EXPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "exports"


def _latest_json_record(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        records = json.load(f)
    return records[-1] if records else None


def _latest_train_run_for_model(model_name: str) -> Optional[dict[str, Any]]:
    if not TRAINING_LOG_PATH.exists():
        return None
    with TRAINING_LOG_PATH.open(encoding="utf-8") as f:
        runs = json.load(f)

    for run in reversed(runs):
        if run.get("model") == model_name:
            return run
    return None


def _to_buy_sell(pred_class: int) -> str:
    return "BUY" if int(pred_class) == 1 else "SELL"


def _render_latest_row_preview(df_processed: pd.DataFrame, feature_columns: list[str]) -> None:
    st.subheader("Latest engineered/scaled record")
    last_row = df_processed.iloc[[-1]].copy()
    cols = ["Date"] + feature_columns + ["Target"]
    st.dataframe(last_row[cols], use_container_width=True, hide_index=True)


def _build_export_markdown(
    *,
    student_name: str,
    student_section: str,
    student_course: str,
    preprocessing_meta: Optional[dict[str, Any]],
    latest_train_runs: dict[str, Optional[dict[str, Any]]],
    latest_evaluation: Optional[dict[str, Any]],
    inference_results: dict[str, dict[str, Any]],
) -> str:
    pre = preprocessing_meta or {}
    eval_rec = latest_evaluation or {}

    def fmt_list(v: Any) -> str:
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)

    lines: list[str] = []
    lines.append("# IS 108 Final Project Report")
    lines.append("")
    lines.append("## Student Information")
    lines.append(f"- Name: {student_name or 'TBD'}")
    lines.append(f"- Section: {student_section or 'TBD'}")
    lines.append(f"- Course: {student_course or 'TBD'}")
    lines.append("")
    lines.append("## Model Prediction Snapshot (Latest Record)")
    lines.append("")

    for model_key in ["knn", "svm", "ann"]:
        tag = inference_results.get(model_key, {}).get("tag", "TBD")
        pred = inference_results.get(model_key, {}).get("pred", None)
        lines.append(f"- {model_key.upper()}: {tag} (pred={pred})")
    lines.append("")

    lines.append("## Preprocessing (Most Recent Run)")
    lines.append("")
    if preprocessing_meta:
        lines.append(f"- Train rows: {pre.get('train_rows')}")
        lines.append(f"- Test rows: {pre.get('test_rows')}")
        lines.append(f"- Train ratio: {pre.get('train_ratio')}")
        lines.append(f"- Scaling method: {pre.get('scaling_method')}")
        lines.append(
            f"- Feature columns: {fmt_list(pre.get('feature_columns', []))}"
        )
    else:
        lines.append("- (Not available)")
    lines.append("")

    lines.append("## Training Hyperparameters (Latest Runs)")
    lines.append("")
    for model_key in ["knn", "svm", "ann"]:
        run = latest_train_runs.get(model_key)
        if not run:
            lines.append(f"- {model_key.upper()}: not trained yet")
            continue
        params = run.get("params", {})
        acc = (run.get("metrics") or {}).get("accuracy")
        lines.append(f"- {model_key.upper()} (test accuracy={acc}):")
        lines.append(f"  - params: {json.dumps(params, default=str)}")
    lines.append("")

    lines.append("## Comparative Evaluation (Latest Evaluation Run)")
    lines.append("")
    if latest_evaluation and eval_rec.get("metrics_by_model"):
        for model_key, m in eval_rec["metrics_by_model"].items():
            lines.append(
                f"- {model_key.upper()}: "
                f"Acc={m.get('accuracy')}, Prec={m.get('precision')}, "
                f"Rec={m.get('recall')}, F1={m.get('f1')}"
            )
    else:
        lines.append("- (Not available)")

    lines.append("")
    lines.append("---")
    lines.append(f"Exported at (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    return "\n".join(lines)


def render_inference_section() -> None:
    st.header("Live Inference (BUY / SELL) & Report Export")
    st.caption(
        "Runs prediction for the latest engineered/scaled record in `processed_dataset.csv` "
        "using the latest hyperparameters from `training_history.json`. "
        "Cached models from Model Training are reused when parameters match."
    )

    bundle = resolve_train_test_data()
    if bundle is None:
        st.warning("Run **Preprocessing** first to generate the processed train/test split.")
        return

    X_train, X_test, y_train, y_test, feature_columns = bundle
    data_key = data_cache_key(X_train, feature_columns)
    processed = load_processed_dataset()
    if processed is None:
        st.warning("`main/data/processed/processed_dataset.csv` not found. Run Preprocessing again.")
        return

    _render_latest_row_preview(processed, feature_columns)

    latest_pre = _latest_json_record(PREPROCESSING_LOG_PATH)
    latest_eval = _latest_json_record(EVALUATION_LOG_PATH)
    latest_runs = {
        "knn": _latest_train_run_for_model("knn"),
        "svm": _latest_train_run_for_model("svm"),
        "ann": _latest_train_run_for_model("ann"),
    }

    st.divider()

    if "inference_results" not in st.session_state:
        st.session_state.inference_results = {}

    col_knn, col_svm, col_ann = st.columns(3)

    def _render_model_panel(model_key: str) -> None:
        st.subheader(f"{model_key.upper()} prediction")
        run = latest_runs[model_key]

        if run is None:
            st.warning(f"{model_key.upper()} not trained yet. Train it in Model Training first.")
            return

        results = st.session_state.get("inference_results", {}).get(model_key)
        if not results:
            st.caption("Run inference to compute BUY/SELL for this model.")
            return

        if results.get("error"):
            st.error(f"Inference error: {results['error']}")
            return

        pred = results.get("pred")
        tag = results.get("tag")
        st.success(f"Signal: {tag} (pred={pred})")

    for col, model_key in zip([col_knn, col_svm, col_ann], ["knn", "svm", "ann"]):
        with col:
            _render_model_panel(model_key)

    if st.button("Run inference for latest record", type="primary"):
        inference_results: dict[str, dict[str, Any]] = {}

        X_latest = processed.iloc[[-1]][feature_columns]

        if latest_runs["knn"] is not None:
            try:
                params = latest_runs["knn"].get("params", {})
                model = get_or_train_knn(
                    X_train,
                    y_train,
                    n_neighbors=int(params.get("n_neighbors", 5)),
                    metric=str(params.get("metric", "euclidean")),
                    data_key=data_key,
                )
                pred = int(model.predict(X_latest)[0])
                inference_results["knn"] = {"pred": pred, "tag": _to_buy_sell(pred)}
            except Exception as exc:
                inference_results["knn"] = {"pred": None, "tag": "ERROR", "error": str(exc)}

        if latest_runs["svm"] is not None:
            try:
                params = latest_runs["svm"].get("params", {})
                model = get_or_train_svm(
                    X_train,
                    y_train,
                    kernel=str(params.get("kernel", "rbf")),
                    C=float(params.get("C", 1.0)),
                    gamma=params.get("gamma", "scale"),
                    data_key=data_key,
                )
                pred = int(model.predict(X_latest)[0])
                inference_results["svm"] = {"pred": pred, "tag": _to_buy_sell(pred)}
            except Exception as exc:
                inference_results["svm"] = {"pred": None, "tag": "ERROR", "error": str(exc)}

        if latest_runs["ann"] is not None:
            try:
                params = latest_runs["ann"].get("params", {})
                model = get_or_train_ann(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    hidden_layers=params.get("hidden_layers", [64, 32]),
                    epochs=int(params.get("epochs", 20)),
                    batch_size=int(params.get("batch_size", 32)),
                    learning_rate=float(params.get("learning_rate", 0.001)),
                    data_key=data_key,
                )
                probs = model.predict(X_latest.values, verbose=0).flatten()
                pred = int(probs[0] >= 0.5)
                inference_results["ann"] = {"pred": pred, "tag": _to_buy_sell(pred)}
            except ImportError as exc:
                inference_results["ann"] = {"pred": None, "tag": "ERROR", "error": str(exc)}
            except Exception as exc:
                inference_results["ann"] = {"pred": None, "tag": "ERROR", "error": str(exc)}

        st.session_state.inference_results = inference_results
        st.success("Inference complete. Scroll down for the export section.")

    if st.session_state.get("inference_results"):
        st.subheader("Inference outputs (BUY / SELL)")
        rows = []
        for model_key in ["knn", "svm", "ann"]:
            r = st.session_state.inference_results.get(model_key)
            if not r:
                continue
            rows.append(
                {
                    "Model": model_key.upper(),
                    "Prediction (0/1)": r.get("pred"),
                    "Signal": r.get("tag"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Academic report export (local file)")

    with st.form("export_form"):
        student_name = st.text_input("Student name")
        student_section = st.text_input("Section")
        student_course = st.text_input("Course")

        export_button = st.form_submit_button("Export report to `main/data/exports/`", type="primary")

    if export_button:
        inference_results = st.session_state.get("inference_results") or {}
        if not inference_results:
            st.warning("Run inference first to include the latest BUY/SELL results in the export.")
            return

        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        out_path = EXPORTS_DIR / f"inference_report_{ts}.md"

        md = _build_export_markdown(
            student_name=student_name,
            student_section=student_section,
            student_course=student_course,
            preprocessing_meta=latest_pre,
            latest_train_runs=latest_runs,
            latest_evaluation=latest_eval,
            inference_results=inference_results,
        )
        out_path.write_text(md, encoding="utf-8")
        st.success(f"Export created: {out_path}")

