Confident Learning Label Auditor
================================

Quickstart
----------

1. Create a virtualenv and install requirements:

   pip install -r confident_learning_app/requirements.txt

2. Run the app:

   python -m confident_learning_app.app

3. Upload a CSV and click "Audit".

   - Text mode: CSV with columns `text` and `label`.
   - Tabular mode: any feature columns + `label`.

Outputs
-------

- Confident Joint heatmap and Uncertainty scatter.
- Issues table ranked by suspicion.
- Exports written to `confident_learning_app/exports/`:
  - `cleaned_*.csv` containing original data + suggested labels and scores.
  - `audit_*.parquet` detailed audit report per-row.

Notes
-----

- Predicted probabilities are cached under `confident_learning_app/cache/`.
- Requires `cleanlab`, `scikit-learn`, `pandas`, `numpy`, `plotly`, `gradio`.
- Image mode is not included in this MVP; extend via `models.py` later.
