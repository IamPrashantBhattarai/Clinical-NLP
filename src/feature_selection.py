"""
feature_selection.py
Purpose: Reduce feature dimensionality to remove noise and improve model
generalization, especially on small clinical datasets.

Methods:
  1. variance_threshold      — drop near-constant features
  2. univariate_selection    — keep top-k by chi2 / f_classif
  3. l1_selection            — sparse Logistic Regression coefficient selection
  4. rfe_selection           — Recursive Feature Elimination (sklearn RFE)
  5. shap_selection          — top-k features by mean |SHAP|
  6. apply_selection         — slice a feature matrix + names by indices
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Variance threshold
# ---------------------------------------------------------------------------

def variance_threshold_selection(
    X,
    feature_names: List[str],
    threshold: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Drop features whose variance is below `threshold`. By default removes
    constant (zero-variance) features.

    Returns
    -------
    (selected_indices, selected_names)
    """
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    mask = selector.get_support()
    indices = np.where(mask)[0]
    names = [feature_names[i] for i in indices if i < len(feature_names)]

    logger.info(
        "Variance threshold (>%.4f): kept %d / %d features",
        threshold, len(indices), X.shape[1],
    )
    return indices, names


# ---------------------------------------------------------------------------
# 2. Univariate selection (chi2 / f_classif / mutual info)
# ---------------------------------------------------------------------------

def univariate_selection(
    X,
    y: np.ndarray,
    feature_names: List[str],
    k: int = 100,
    score_func: str = "f_classif",
) -> Tuple[np.ndarray, List[str]]:
    """
    Keep the top-k features by univariate statistical test.

    Parameters
    ----------
    score_func : str
        'chi2' (non-negative features only), 'f_classif', or 'mutual_info'.
    k : int
        Number of features to keep. If k >= n_features, returns all.
    """
    from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

    funcs = {
        "chi2": chi2,
        "f_classif": f_classif,
        "mutual_info": mutual_info_classif,
    }
    if score_func not in funcs:
        raise ValueError(f"Unknown score_func: {score_func}")

    n_features = X.shape[1]
    k_eff = min(k, n_features)

    selector = SelectKBest(score_func=funcs[score_func], k=k_eff)
    selector.fit(X, y)
    mask = selector.get_support()
    indices = np.where(mask)[0]
    names = [feature_names[i] for i in indices if i < len(feature_names)]

    logger.info(
        "Univariate (%s): kept top %d / %d features",
        score_func, len(indices), n_features,
    )
    return indices, names


# ---------------------------------------------------------------------------
# 3. L1-based selection (sparse logistic regression)
# ---------------------------------------------------------------------------

def l1_selection(
    X,
    y: np.ndarray,
    feature_names: List[str],
    C: float = 1.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Use a sparse L1-penalized Logistic Regression to identify features with
    non-zero coefficients.

    Parameters
    ----------
    C : float
        Inverse regularization strength. Smaller = sparser.
    """
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=C,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=1,
    )
    lr.fit(X, y)

    coefs = lr.coef_.ravel()
    indices = np.where(np.abs(coefs) > 1e-8)[0]
    if len(indices) == 0:
        logger.warning("L1 selection eliminated all features — falling back to top 50 by |coef|.")
        indices = np.argsort(-np.abs(coefs))[:50]

    names = [feature_names[i] for i in indices if i < len(feature_names)]
    logger.info(
        "L1 (C=%.3f): kept %d / %d non-zero features",
        C, len(indices), X.shape[1],
    )
    return indices, names


# ---------------------------------------------------------------------------
# 4. Recursive Feature Elimination
# ---------------------------------------------------------------------------

def rfe_selection(
    X,
    y: np.ndarray,
    feature_names: List[str],
    n_features_to_select: int = 50,
    step: float = 0.1,
    random_seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Recursive Feature Elimination using a Logistic Regression base estimator.

    Parameters
    ----------
    n_features_to_select : int
        Target number of features.
    step : float
        Fraction of features to remove at each iteration.
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    n_features = X.shape[1]
    n_target = min(n_features_to_select, n_features)

    base = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=500,
        class_weight="balanced",
        random_state=random_seed,
    )

    # Convert sparse → dense if small enough; RFE works with sparse for some estimators
    X_in = X.toarray() if issparse(X) and n_features < 5000 else X

    selector = RFE(estimator=base, n_features_to_select=n_target, step=step)
    selector.fit(X_in, y)
    indices = np.where(selector.support_)[0]
    names = [feature_names[i] for i in indices if i < len(feature_names)]

    logger.info(
        "RFE: kept %d / %d features (step=%.2f)",
        len(indices), n_features, step,
    )
    return indices, names


# ---------------------------------------------------------------------------
# 5. SHAP-based selection
# ---------------------------------------------------------------------------

def shap_selection(
    model,
    X,
    feature_names: List[str],
    model_name: str = "",
    top_k: int = 50,
    max_samples: int = 200,
) -> Tuple[np.ndarray, List[str]]:
    """
    Select the top-k features by mean absolute SHAP value.

    Requires a fitted model. Reuses src.explainability.compute_shap_values.

    Parameters
    ----------
    top_k : int
        Number of features to keep.
    max_samples : int
        Cap on samples used to estimate SHAP (for speed).
    """
    from src.explainability import compute_shap_values

    out = compute_shap_values(
        model, X, X_background=X,
        model_name=model_name, max_samples=max_samples,
    )
    shap_values = out["shap_values"]
    mean_abs = np.abs(shap_values).mean(axis=0)

    n_features = X.shape[1]
    k_eff = min(top_k, n_features)
    indices = np.argsort(-mean_abs)[:k_eff]
    indices.sort()  # keep original column order
    names = [feature_names[i] for i in indices if i < len(feature_names)]

    logger.info(
        "SHAP: kept top %d / %d features (model=%s)",
        len(indices), n_features, model_name,
    )
    return indices, names


# ---------------------------------------------------------------------------
# 6. Apply selection
# ---------------------------------------------------------------------------

def apply_selection(X, indices: np.ndarray):
    """Slice a feature matrix (sparse or dense) by column indices."""
    if issparse(X):
        return X[:, indices]
    return X[:, indices]


# ---------------------------------------------------------------------------
# 7. End-to-end selector
# ---------------------------------------------------------------------------

def select_features(
    X,
    y: np.ndarray,
    feature_names: List[str],
    method: str = "univariate",
    **kwargs,
) -> dict:
    """
    Run feature selection by name and return a dict with the reduced matrix.

    Parameters
    ----------
    method : str
        One of: 'variance', 'univariate', 'l1', 'rfe', 'shap'.
    **kwargs
        Method-specific parameters (k, C, n_features_to_select, top_k, ...).
        For 'shap', must include `model` and `model_name`.

    Returns
    -------
    dict with keys:
        'X_selected'    — reduced feature matrix
        'indices'       — selected column indices
        'names'         — selected feature names
        'method'        — method used
        'n_before' / 'n_after'
    """
    n_before = X.shape[1]

    if method == "variance":
        indices, names = variance_threshold_selection(
            X, feature_names, threshold=kwargs.get("threshold", 0.0),
        )
    elif method == "univariate":
        indices, names = univariate_selection(
            X, y, feature_names,
            k=kwargs.get("k", 100),
            score_func=kwargs.get("score_func", "f_classif"),
        )
    elif method == "l1":
        indices, names = l1_selection(
            X, y, feature_names,
            C=kwargs.get("C", 1.0),
            random_seed=kwargs.get("random_seed", 42),
        )
    elif method == "rfe":
        indices, names = rfe_selection(
            X, y, feature_names,
            n_features_to_select=kwargs.get("n_features_to_select", 50),
            step=kwargs.get("step", 0.1),
            random_seed=kwargs.get("random_seed", 42),
        )
    elif method == "shap":
        model = kwargs.get("model")
        if model is None:
            raise ValueError("SHAP selection requires a fitted 'model' kwarg.")
        indices, names = shap_selection(
            model, X, feature_names,
            model_name=kwargs.get("model_name", ""),
            top_k=kwargs.get("top_k", 50),
            max_samples=kwargs.get("max_samples", 200),
        )
    else:
        raise ValueError(f"Unknown selection method: '{method}'")

    X_selected = apply_selection(X, indices)
    n_after = X_selected.shape[1]

    logger.info(
        "Feature selection (%s): %d -> %d features (%.0f%% reduction)",
        method, n_before, n_after, 100 * (1 - n_after / max(n_before, 1)),
    )

    return {
        "X_selected": X_selected,
        "indices": indices,
        "names": names,
        "method": method,
        "n_before": n_before,
        "n_after": n_after,
    }
