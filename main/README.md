# Dogecoin BI Application

Streamlit **Financial Intelligence** dashboard for IS 108 (Caraga State University, SY 2025–2026): binary directional trading signals for Dogecoin (DOGE) using KNN, SVM, and ANN. All pipeline state is stored under `main/data/` as local CSV/JSON files (no database).

Project operating docs (tasks, architecture) live in [`../paws022/`](../paws022/).

GitHub: [verdenef/marketTrendPrediction](https://github.com/verdenef/marketTrendPrediction) (`master`).

---

## Requirements

| Requirement | Notes |
|-------------|--------|
| **Python 3.12** | Required for TensorFlow/Keras (ANN). Python 3.14 cannot install TensorFlow. |
| **Windows / macOS / Linux** | Commands below use PowerShell; adapt paths for your OS. |

---

## Environment setup

From the repository root (`c:\dev\is108final` or your clone):

```powershell
cd c:\dev\is108final
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r main\requirements.txt
```

Core packages (see `main/requirements.txt`): `streamlit`, `pandas`, `numpy`, `scikit-learn`, `openpyxl`, `matplotlib`, `tensorflow`.

If you previously used a Python 3.14 virtualenv, **do not** use it for ANN. Stop Streamlit, activate `.venv312`, and reinstall as above.

Verify TensorFlow (optional):

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Run the app

```powershell
.\.venv312\Scripts\Activate.ps1
streamlit run main\src\app.py
```

The app opens in your browser (default `http://localhost:8501`).

---

## End-to-end demo (grading / submission)

Follow the sidebar **Grader demo** checklist or these steps:

| Step | Tab | Action |
|------|-----|--------|
| 1 | **Dataset** | Click **Load project Dogecoin dataset** (bundled `main/data/coin_Dogecoin.csv`) or upload CSV/XLSX with OHLCV columns. |
| 2 | **Preprocessing** | Click **Run full preprocessing pipeline** (indicators, scaling, chronological train/test split). |
| 3 | **Model Training** | Click **Train all models (demo presets)** — K=5 KNN, RBF SVM, ANN `64,32` / 20 epochs — or train each model individually. |
| 4 | **Evaluation** | Click **Run comparative evaluation** (metrics table + confusion matrices). Uses cached models when parameters match training. |
| 5 | **Live Inference** | Click **Run inference for latest record** (BUY/SELL per model), then **Export report** to `main/data/exports/`. |

**Target meaning:** `Target = 1` → next-day close up (**BUY**); `Target = 0` → down (**SELL**).

---

## Data format

Required columns (extra columns such as `SNo`, `Symbol`, `Marketcap` are ignored):

| Date | Open | High | Low | Close | Volume |

Supported uploads: `.csv`, `.xlsx`.

---

## Local storage (ADR-003: file-based only)

| Path | Purpose |
|------|---------|
| `main/data/coin_Dogecoin.csv` | Bundled historical DOGE data |
| `main/data/active_dataset.csv` | Last loaded dataset |
| `main/data/processed/processed_dataset.csv` | Engineered + scaled features |
| `main/data/logs/dataset_imports.json` | Dataset import log |
| `main/data/logs/preprocessing_runs.json` | Preprocessing run log |
| `main/data/logs/training_history.json` | KNN / SVM / ANN training log |
| `main/data/logs/evaluation_history.json` | Comparative evaluation log |
| `main/data/exports/` | Exported markdown reports |

Logs and exports are gitignored; regenerate by running the pipeline in the UI.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| ANN import / TensorFlow error | Use Python **3.12** and `.venv312`; see environment setup above. |
| “Preprocessed data not found” | Complete **Preprocessing** before training or evaluation. |
| Slow first Evaluation / Inference run | Normal on first visit; subsequent runs reuse `st.session_state` model cache unless preprocessing or hyperparameters change. |
| Streamlit file lock on venv | Stop Streamlit before recreating a virtual environment. |

---

## Source layout

```
main/src/
  app.py                 # Streamlit entry
  components/            # Dataset, preprocessing, training, evaluation, inference UI
  models/                # KNN, SVM, ANN wrappers
  utils/                 # Preprocessing, file storage, train data, model cache
```
