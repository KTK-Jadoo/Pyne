from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from confident_learning_app.models import (
    build_tabular_model,
    build_text_model,
    crossval_predict_proba,
)
from confident_learning_app.utils import (
    compute_df_hash,
    export_audit,
    load_pred_probs,
    save_pred_probs,
    split_feature_columns,
)


# Optional imports with graceful fallback across cleanlab versions
try:  # cleanlab v2 API
    from cleanlab.rank import order_label_issues, get_label_quality_scores
except Exception:  # pragma: no cover - fallback
    order_label_issues = None
    get_label_quality_scores = None

try:
    from cleanlab.filter import find_label_issues
except Exception:  # pragma: no cover - fallback
    find_label_issues = None

try:
    from cleanlab.count import compute_confident_joint
except Exception:  # pragma: no cover - fallback
    compute_confident_joint = None


@dataclass
class AuditResults:
    df_augmented: pd.DataFrame
    issues_df: pd.DataFrame
    pred_probs: np.ndarray
    cj: Optional[np.ndarray]
    classes: List[str]
    self_confidence: np.ndarray
    alternative_prob: np.ndarray
    ranked_indices: np.ndarray
    exports: Optional[Dict[str, str]]


def _quality_scores(labels_int: np.ndarray, psx: np.ndarray) -> Optional[np.ndarray]:
    if get_label_quality_scores is not None:
        try:
            return get_label_quality_scores(labels_int, psx)
        except Exception:
            return None
    return None


def _rank_issues(labels_int: np.ndarray, psx: np.ndarray) -> np.ndarray:
    n = len(labels_int)
    # Preferred API
    if order_label_issues is not None:
        try:
            ranked = order_label_issues(labels=labels_int, pred_probs=psx)
            return np.asarray(ranked, dtype=int)
        except Exception:
            pass
    # Fallback API across versions
    if find_label_issues is not None:
        try:
            out = find_label_issues(labels=labels_int, pred_probs=psx)
            out = np.asarray(out)
            if out.dtype == bool:
                return np.where(out)[0]
            return out.astype(int)
        except Exception:
            pass
    # Final fallback: rank by ascending self-confidence
    self_conf = psx[np.arange(n), labels_int]
    return np.argsort(self_conf)


def _compute_self_and_alt_probs(labels_int: np.ndarray, psx: np.ndarray) -> (np.ndarray, np.ndarray):
    n = len(labels_int)
    self_conf = psx[np.arange(n), labels_int]
    alt_prob = np.copy(psx)
    alt_prob[np.arange(n), labels_int] = -np.inf
    alt_prob = alt_prob.max(axis=1)
    return self_conf, alt_prob


def _suggested_labels(psx: np.ndarray, classes: List[str]) -> List[str]:
    preds_int = psx.argmax(axis=1)
    return [classes[i] for i in preds_int]


def audit_tabular_text(
    df: pd.DataFrame,
    task: str,
    label_col: str,
    text_col: Optional[str] = None,
    cv: int = 5,
    use_cache: bool = True,
    export: bool = True,
) -> AuditResults:
    """Run Confident Learning-based audit for tabular or text data.

    Args:
        df: Input DataFrame.
        task: "text" or "tabular".
        label_col: Name of the label column.
        text_col: Name of the text column (for text task).
        cv: Number of cross-validation folds.
        use_cache: If True, load/save predicted probabilities from cache.
        export: If True, write cleaned CSV and audit report under exports/.
    """
    if task not in {"text", "tabular"}:
        raise ValueError("task must be 'text' or 'tabular'")
    if task == "text" and not text_col:
        raise ValueError("text_col is required for text task")

    df_local = df.copy()
    y_raw = df_local[label_col].astype(str).fillna("")
    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)
    classes = le.classes_.tolist()

    # Build features/model
    if task == "text":
        X = df_local[text_col].astype(str).fillna(" ")
        model = build_text_model()
        cache_key = f"{task}_{compute_df_hash(df_local[[text_col, label_col]])}_cv{cv}"
    else:
        numeric_cols, categorical_cols = split_feature_columns(df_local, label_col)
        X = df_local.drop(columns=[label_col])
        model = build_tabular_model(numeric_cols, categorical_cols)
        cache_key = f"{task}_{compute_df_hash(df_local)}_cv{cv}"

    # Predicted probabilities (out-of-fold)
    psx: Optional[np.ndarray] = None
    if use_cache:
        psx = load_pred_probs(cache_key)
    if psx is None:
        psx = crossval_predict_proba(model, X, y, cv=cv)
        if use_cache:
            save_pred_probs(cache_key, psx)

    # Confident learning artifacts
    cj = None
    if compute_confident_joint is not None:
        try:
            cj = compute_confident_joint(y=y, pred_probs=psx)
        except Exception:
            cj = None

    self_conf, alt_prob = _compute_self_and_alt_probs(y, psx)
    quality = _quality_scores(y, psx)
    ranked = _rank_issues(y, psx)

    # Build issues table
    suggested = _suggested_labels(psx, classes)
    given = y_raw.tolist()
    is_issue = np.array([
        (g != s) and (sc < ap)
        for g, s, sc, ap in zip(given, suggested, self_conf, alt_prob)
    ])

    issues_df = pd.DataFrame(
        {
            "index": np.arange(len(df_local)),
            "given_label": given,
            "predicted_label": [classes[i] for i in psx.argmax(axis=1)],
            "suggested_label": suggested,
            "self_confidence": self_conf,
            "alternative_prob": alt_prob,
            "quality_score": quality if quality is not None else -self_conf,  # lower is worse
            "is_label_issue": is_issue,
        }
    )
    # rank: 0 is most suspicious (full-length rank vector)
    n = len(df_local)
    rank_vec = np.full(n, fill_value=n, dtype=int)
    if ranked.dtype == bool and ranked.shape == (n,):
        idx = np.where(ranked)[0]
        rank_vec[idx] = np.arange(len(idx))
        ranked_indices = idx
    else:
        ranked_indices = ranked.astype(int)
        rank_vec[ranked_indices] = np.arange(len(ranked_indices))
    issues_df["rank"] = rank_vec
    issues_df.sort_values("rank", inplace=True)
    issues_df.reset_index(drop=True, inplace=True)

    df_aug = df_local.copy()
    df_aug["predicted_label"] = issues_df.sort_values("index")["predicted_label"].values
    df_aug["suggested_label"] = issues_df.sort_values("index")["suggested_label"].values
    df_aug["self_confidence"] = issues_df.sort_values("index")["self_confidence"].values
    df_aug["is_label_issue"] = issues_df.sort_values("index")["is_label_issue"].values

    exports = None
    if export:
        arts = export_audit(df_local, issues_df, cache_key)
        exports = {"cleaned_csv": arts.cleaned_csv, "audit_parquet": arts.audit_parquet}

    return AuditResults(
        df_augmented=df_aug,
        issues_df=issues_df,
        pred_probs=psx,
        cj=cj,
        classes=classes,
        self_confidence=self_conf,
        alternative_prob=alt_prob,
        ranked_indices=ranked_indices,
        exports=exports,
    )
