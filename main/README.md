# Dogecoin BI Application

Streamlit app for IS 108 — Dogecoin market trend / trading signal prediction.

POS (tasks, architecture, docs) lives in [`../paws022/`](../paws022/).

## Setup

```powershell
cd c:\dev\is108final
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r main\requirements.txt
```

Use **Python 3.12** for the full app (including ANN). TensorFlow does not install on Python 3.14.

If an old `.venv` was created with 3.14, stop Streamlit, then either activate `.venv312` as above or replace `.venv` after closing the app:

```powershell
py -3.12 -m venv .venv --clear
.\.venv\Scripts\Activate.ps1
pip install -r main\requirements.txt
```

## Run

```powershell
streamlit run main\src\app.py
```

1. **Dataset** → **Load project Dogecoin dataset** (or upload a file).
2. **Preprocessing** → **Run full preprocessing pipeline**.
3. **Model Training** → **Train KNN** / **Train SVM** / **Train ANN** (see loss chart for ANN).

## Data format

Required columns (extras like `SNo`, `Symbol`, `Marketcap` are ignored):

| Date | Open | High | Low | Close | Volume |

Supported upload formats: `.csv`, `.xlsx`.

## Local storage

| File | Purpose |
|------|---------|
| `data/coin_Dogecoin.csv` | Bundled historical DOGE data |
| `data/active_dataset.csv` | Last loaded dataset (auto-saved) |
| `data/logs/dataset_imports.json` | Import history |
| `data/processed/processed_dataset.csv` | Engineered + scaled features |
| `data/logs/preprocessing_runs.json` | Preprocessing run history |
