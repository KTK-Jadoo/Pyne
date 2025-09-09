import os
from typing import Optional, Tuple
import re

import gradio as gr
import numpy as np
import pandas as pd

from confident_learning_app.core import audit_tabular_text
from confident_learning_app.utils import (
    DATA_DIR,
    EXPORTS_DIR,
    compute_df_hash,
    ensure_dirs,
    guess_label_column,
    guess_text_column,
)
from confident_learning_app.viz import confident_joint_heatmap, uncertainty_scatter


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", " ".join(s.split()).strip().lower())


def _canonicalize_column(cols: list[str], candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    if candidate in cols:
        return candidate
    norm_map = {_norm(c): c for c in cols}
    return norm_map.get(_norm(candidate))


def _detect_task_and_cols(
    df: pd.DataFrame, task_choice: str, label_col: Optional[str], text_col: Optional[str]
) -> Tuple[str, Optional[str], Optional[str]]:
    cols = df.columns.tolist()
    # Prefer provided label (canonicalized); otherwise guess
    provided_label = _canonicalize_column(cols, (label_col or "").strip())
    label = provided_label or guess_label_column(cols)
    if task_choice == "auto":
        tcol_guess = guess_text_column(cols)
        if tcol_guess is not None:
            task = "text"
            tcol = _canonicalize_column(cols, text_col) or tcol_guess
        else:
            task = "tabular"
            tcol = None
    else:
        task = task_choice
        tcol = _canonicalize_column(cols, text_col) if task == "text" else None
    return task, label, tcol


def run_audit(
    file: gr.File | str | None,
    dataset_type: str,
    label_col: Optional[str],
    text_col: Optional[str],
    cv_folds: int,
):
    if file is None:
        raise gr.Error("Please upload a CSV file.")
    ensure_dirs()
    # Support both filepath str and UploadedFile object
    if isinstance(file, (str, os.PathLike)):
        path = str(file)
    else:
        path = file.name  # type: ignore[attr-defined]
    df = pd.read_csv(path)
    task, label_col, tcol = _detect_task_and_cols(df, dataset_type, label_col, text_col)

    if not label_col:
        sample_cols = ", ".join(map(str, df.columns[:10]))
        raise gr.Error(
            "Select a Label column from the dropdown after upload. "
            f"Available columns: {sample_cols}"
        )
    if task == "text" and not tcol:
        raise gr.Error("Select a Text column from the dropdown for text datasets.")

    results = audit_tabular_text(
        df=df,
        task=task,
        label_col=label_col,
        text_col=tcol,
        cv=cv_folds,
        use_cache=True,
        export=True,
    )

    # Visuals
    heatmap = confident_joint_heatmap(results.cj, results.classes)
    scatter = uncertainty_scatter(results.self_confidence, results.alternative_prob, results.issues_df["is_label_issue"].values)

    # Top suspicious examples
    top_issues = results.issues_df.head(200)

    clean_path = results.exports.get("cleaned_csv") if results.exports else None
    report_path = results.exports.get("audit_parquet") if results.exports else None

    return heatmap, scatter, top_issues, clean_path, report_path


def _filepath_from_file(file: gr.File | str | None) -> str:
    if file is None:
        raise gr.Error("Please upload a CSV file.")
    if isinstance(file, (str, os.PathLike)):
        return str(file)
    return file.name  # type: ignore[attr-defined]


def parse_columns(file: gr.File | str | None, current_dataset_type: str):
    """Read only the header to populate column selectors and optionally set dataset type."""
    try:
        path = _filepath_from_file(file)
        # Read a small sample to infer types/lengths for heuristics
        df_head = pd.read_csv(path, nrows=0)
        df_sample = pd.read_csv(path, nrows=50)
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), f"Failed to read CSV header: {e}"

    cols = df_head.columns.tolist()
    label_guess = guess_label_column(cols) or None
    if label_guess is None:
        label_keywords = [
            "label",
            "target",
            "class",
            "category",
            "stress",
            "index",
            "score",
            "rating",
            "rate",
        ]
        label_matches = [c for c in cols if any(k in c.lower() for k in label_keywords)]
        if label_matches:
            # Often the rightmost column is the target
            label_guess = label_matches[-1]

    # Heuristic for text column: look for a single long string column
    text_guess = None
    strong_text_candidates = []
    if not df_sample.empty:
        for c in cols:
            s = df_sample[c]
            if not pd.api.types.is_object_dtype(s):
                continue
            values = s.dropna().astype(str)
            if values.empty:
                continue
            avg_len = values.str.len().mean()
            frac_with_space = (values.str.contains(r"\s")).mean()
            # Consider as strong text if it looks like free-form sentences
            if avg_len >= 20 and frac_with_space >= 0.5:
                strong_text_candidates.append((avg_len, c))
        strong_text_candidates.sort(reverse=True)
        if len(strong_text_candidates) == 1:
            text_guess = strong_text_candidates[0][1]

    ds_update = gr.update()
    if current_dataset_type == "auto":
        # Only flip to text if we found exactly one strong candidate
        ds_update = gr.update(value="text" if text_guess else "tabular")

    info = f"Detected {len(cols)} columns: " + ", ".join(map(str, cols[:30]))
    if len(cols) > 30:
        info += ", ..."

    return (
        gr.update(choices=cols, value=label_guess),
        gr.update(choices=cols, value=text_guess),
        ds_update,
        info,
    )


def app() -> gr.Blocks:
    with gr.Blocks(title="Confident Learning Label Auditor") as demo:
        gr.Markdown("""
        # Confident Learning Label Auditor
        Upload a CSV (text: columns `text` and `label`; tabular: any features + `label`).
        Choose dataset type or leave as Auto. Then click Audit.
        """)

        with gr.Row():
            file = gr.File(label="CSV file", file_types=[".csv"], type="filepath")
        with gr.Row():
            dataset_type = gr.Radio(
                ["auto", "text", "tabular"],
                value="auto",
                label="Dataset type",
            )
            label_col = gr.Dropdown(choices=[], label="Label column", interactive=True)
            text_col = gr.Dropdown(choices=[], label="Text column (for text)", interactive=True)
            cv_folds = gr.Slider(3, 10, value=5, step=1, label="CV folds")

        columns_info = gr.Markdown(visible=True)

        run_btn = gr.Button("Audit")

        with gr.Tabs():
            with gr.TabItem("Heatmap"):
                heatmap = gr.Plot()
            with gr.TabItem("Scatter"):
                scatter = gr.Plot()
            with gr.TabItem("Issues Table"):
                issues = gr.Dataframe()

        with gr.Row():
            cleaned_csv = gr.File(label="Cleaned CSV", interactive=False)
            audit_report = gr.File(label="Audit report (Parquet)", interactive=False)

        file.change(
            fn=parse_columns,
            inputs=[file, dataset_type],
            outputs=[label_col, text_col, dataset_type, columns_info],
        )

        run_btn.click(
            fn=run_audit,
            inputs=[file, dataset_type, label_col, text_col, cv_folds],
            outputs=[heatmap, scatter, issues, cleaned_csv, audit_report],
        )

    return demo


if __name__ == "__main__":
    app().launch()
