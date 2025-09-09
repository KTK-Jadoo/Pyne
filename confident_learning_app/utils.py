import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


APP_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_ROOT, "data")
CACHE_DIR = os.path.join(APP_ROOT, "cache")
EXPORTS_DIR = os.path.join(APP_ROOT, "exports")


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(EXPORTS_DIR, exist_ok=True)


def compute_df_hash(df: pd.DataFrame) -> str:
    """Compute a stable hash for a DataFrame to use for caching.

    Uses CSV bytes of sorted columns (by name) for reproducibility.
    """
    cols_sorted = sorted(df.columns.tolist())
    df_sorted = df[cols_sorted]
    # Limit size for huge files: sample rows deterministically if > 100k rows
    if len(df_sorted) > 100_000:
        df_sorted = df_sorted.sample(n=100_000, random_state=42)
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    return hashlib.md5(csv_bytes).hexdigest()  # nosec - not for security, just caching


def cache_path(kind: str, key: str) -> str:
    """Return a path inside cache for a given artifact kind and key."""
    safe_key = key.replace(os.sep, "_")[:128]
    return os.path.join(CACHE_DIR, f"{kind}_{safe_key}")


def save_pred_probs(key: str, pred_probs: np.ndarray) -> str:
    ensure_dirs()
    path = cache_path("pred_probs", key) + ".npz"
    np.savez_compressed(path, pred_probs=pred_probs)
    return path


def load_pred_probs(key: str) -> Optional[np.ndarray]:
    path = cache_path("pred_probs", key) + ".npz"
    if os.path.exists(path):
        with np.load(path) as data:
            return data["pred_probs"]
    return None


def guess_label_column(columns: List[str]) -> Optional[str]:
    candidates = [
        "label",
        "labels",
        "target",
        "class",
        "category",
        "y",
    ]
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def guess_text_column(columns: List[str]) -> Optional[str]:
    candidates = ["text", "sentence", "content", "body", "review", "message"]
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def split_feature_columns(df: pd.DataFrame, label_col: str) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) for tabular data."""
    feature_cols = [c for c in df.columns if c != label_col]
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


@dataclass
class ExportArtifacts:
    cleaned_csv: str
    audit_parquet: str  # may be .parquet or .csv if parquet engine unavailable


def export_audit(
    original_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    key: str,
) -> ExportArtifacts:
    """Export cleaned dataset and audit report.

    The cleaned CSV includes suggested labels in `suggested_label` and does not overwrite
    the original `label` column. The audit report is stored as Parquet.
    """
    ensure_dirs()
    # Merge issues back to original df by index
    merged = original_df.copy()
    for col in [
        "given_label",
        "predicted_label",
        "suggested_label",
        "self_confidence",
        "alternative_prob",
        "quality_score",
        "is_label_issue",
        "rank",
    ]:
        if col in issues_df.columns:
            merged[col] = issues_df[col]

    cleaned_csv = os.path.join(EXPORTS_DIR, f"cleaned_{key}.csv")
    audit_parquet = os.path.join(EXPORTS_DIR, f"audit_{key}.parquet")
    merged.to_csv(cleaned_csv, index=False)
    try:
        issues_df.to_parquet(audit_parquet, index=False)
    except Exception:
        # Fallback if pyarrow/fastparquet not installed
        audit_parquet = os.path.join(EXPORTS_DIR, f"audit_{key}.csv")
        issues_df.to_csv(audit_parquet, index=False)
    return ExportArtifacts(cleaned_csv=cleaned_csv, audit_parquet=audit_parquet)
