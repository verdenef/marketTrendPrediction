# IS108 Final Project: Dogecoin Trend BI App

File-based Streamlit Business Intelligence application for IS 108 (Caraga State University, SY 2025-2026) that predicts binary Dogecoin market direction signals (**BUY/SELL**) using KNN, SVM, and ANN.

## Repository layout

- `main/` — application source code, requirements, and runbook
- `paws022/` — local planning/docs workspace (not part of deployed app)

## Quick start

Use Python 3.12 for ANN/TensorFlow support:

```powershell
cd c:\dev\is108final
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r main\requirements.txt
streamlit run main\src\app.py
```

## Full documentation

See `main/README.md` for:
- complete grading/demo flow (Dataset -> Preprocessing -> Training -> Evaluation -> Live Inference)
- local file storage/log paths under `main/data/`
- troubleshooting notes (including TensorFlow + Python version constraints)