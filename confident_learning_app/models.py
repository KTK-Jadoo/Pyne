from typing import List, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_text_model() -> Pipeline:
    """TF-IDF + Logistic Regression pipeline for text classification."""
    text_clf = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=1,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return text_clf


def build_tabular_model(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    """ColumnTransformer + Logistic Regression for tabular data."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return model


def crossval_predict_proba(
    estimator: Pipeline, X, y: np.ndarray, cv: int = 5, random_state: int = 42
) -> np.ndarray:
    """Out-of-fold predicted probabilities using StratifiedKFold."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    psx = cross_val_predict(
        estimator,
        X,
        y,
        cv=skf,
        method="predict_proba",
        verbose=0,
    )
    return psx
