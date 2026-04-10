"""
explainability.py
Purpose: Generate SHAP-based explanations for clinical readmission predictions.

Provides:
  - Global feature importance via SHAP values
  - Per-patient (local) explanations for individual predictions
  - Support for tree-based (XGBoost, LightGBM, RF) and linear models
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Explainer factory
# ---------------------------------------------------------------------------

def _to_dense(X):
    """Convert sparse matrix to dense if needed."""
    if issparse(X):
        return X.toarray()
    return X


def get_explainer(model, X_background, model_name: str = ""):
    """
    Build an appropriate SHAP explainer for the given model.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    X_background : array-like
        Background data for the explainer (use train set or a subsample).
    model_name : str
        One of: logistic_regression, random_forest, xgboost, lightgbm.

    Returns
    -------
    shap.Explainer
    """
    import shap

    name = model_name.lower()

    # Tree-based: use TreeExplainer (fast, exact)
    if name in ("xgboost", "lightgbm", "random_forest"):
        logger.info("Using TreeExplainer for %s", model_name)
        return shap.TreeExplainer(model)

    # Linear models: use LinearExplainer
    if name == "logistic_regression":
        logger.info("Using LinearExplainer for %s", model_name)
        X_bg = _to_dense(X_background)
        return shap.LinearExplainer(model, X_bg)

    # Fallback: KernelExplainer (slow but model-agnostic)
    logger.info("Using KernelExplainer (model-agnostic) for %s", model_name)
    X_bg = _to_dense(X_background)
    # Subsample background to keep KernelExplainer tractable
    if X_bg.shape[0] > 50:
        idx = np.random.choice(X_bg.shape[0], 50, replace=False)
        X_bg = X_bg[idx]
    return shap.KernelExplainer(model.predict_proba, X_bg)


# ---------------------------------------------------------------------------
# 2. Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X_explain,
    X_background=None,
    model_name: str = "",
    max_samples: int = 100,
) -> dict:
    """
    Compute SHAP values for a set of samples.

    Parameters
    ----------
    model : fitted classifier
    X_explain : array-like
        Samples to explain (typically test set).
    X_background : array-like, optional
        Background data for the explainer (defaults to X_explain).
    model_name : str
    max_samples : int
        Cap on samples to explain (SHAP can be slow).

    Returns
    -------
    dict with keys:
        'shap_values'  — np.ndarray, shape (n_samples, n_features)
        'base_value'   — float, expected model output
        'X'            — dense feature matrix used for explanation
        'explainer'    — fitted SHAP explainer
    """
    import shap

    # Cap samples
    n = X_explain.shape[0]
    if n > max_samples:
        logger.info("Subsampling explanation set: %d -> %d", n, max_samples)
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        if issparse(X_explain):
            X_explain = X_explain[idx]
        else:
            X_explain = X_explain[idx]

    if X_background is None:
        X_background = X_explain

    explainer = get_explainer(model, X_background, model_name=model_name)
    X_dense = _to_dense(X_explain)

    logger.info("Computing SHAP values for %d samples ...", X_dense.shape[0])
    shap_out = explainer.shap_values(X_dense)

    # SHAP returns a list for multi-class; pick positive class (index 1)
    if isinstance(shap_out, list):
        shap_values = shap_out[1] if len(shap_out) > 1 else shap_out[0]
    else:
        shap_values = shap_out

    # Some explainers return 3D (n_samples, n_features, n_classes) for binary
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Base value
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.atleast_1d(base_value)[-1])
    else:
        base_value = float(base_value)

    logger.info("SHAP values shape: %s, base value: %.4f", shap_values.shape, base_value)

    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "X": X_dense,
        "explainer": explainer,
    }


# ---------------------------------------------------------------------------
# 3. Global feature importance from SHAP
# ---------------------------------------------------------------------------

def shap_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Aggregate SHAP values into a global feature importance ranking.

    Importance = mean(|SHAP value|) per feature.

    Returns
    -------
    pd.DataFrame with columns ['feature', 'mean_abs_shap', 'mean_shap'],
    sorted descending by mean_abs_shap, top_n rows.
    """
    n_features = shap_values.shape[1]
    n = min(n_features, len(feature_names))

    mean_abs = np.abs(shap_values[:, :n]).mean(axis=0)
    mean_signed = shap_values[:, :n].mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names[:n],
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4. Per-patient (local) explanations
# ---------------------------------------------------------------------------

def explain_patient(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    patient_idx: int,
    base_value: float,
    top_n: int = 10,
) -> dict:
    """
    Build a local explanation for a single patient.

    Returns the top features pushing the prediction toward / away from
    readmission, plus the predicted probability and base value.

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_samples, n_features)
    X : np.ndarray, shape (n_samples, n_features)
        The feature values used to compute SHAP.
    feature_names : list of str
    patient_idx : int
        Index into the SHAP/X arrays.
    base_value : float
        Expected model output (baseline).
    top_n : int

    Returns
    -------
    dict with keys:
        'patient_idx', 'base_value', 'predicted_logit', 'top_features' (DataFrame)
    """
    n_features = shap_values.shape[1]
    n = min(n_features, len(feature_names))

    patient_shap = shap_values[patient_idx, :n]
    patient_x = X[patient_idx, :n]

    df = pd.DataFrame({
        "feature": feature_names[:n],
        "value": patient_x,
        "shap": patient_shap,
        "abs_shap": np.abs(patient_shap),
    })
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    df["direction"] = df["shap"].apply(lambda x: "increases risk" if x > 0 else "decreases risk")

    predicted_logit = base_value + patient_shap.sum()

    return {
        "patient_idx": int(patient_idx),
        "base_value": float(base_value),
        "predicted_logit": float(predicted_logit),
        "top_features": df,
    }


# ---------------------------------------------------------------------------
# 5. End-to-end SHAP analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(
    prediction_results: dict,
    feature_sets: dict,
    config: Optional[dict] = None,
    n_explain: int = 100,
    n_patients_to_show: int = 3,
) -> dict:
    """
    Run SHAP explanation on the best model from a prediction pipeline run.

    Parameters
    ----------
    prediction_results : dict
        Output of predict.run_prediction_pipeline().
    feature_sets : dict
        Output of feature_engineer.build_feature_sets().
    config : dict, optional
    n_explain : int
        Max test samples to explain.
    n_patients_to_show : int
        Number of patient-level explanations to return.

    Returns
    -------
    dict with keys:
        'model_name', 'feature_type', 'shap_values', 'base_value',
        'X_test', 'feature_names', 'global_importance', 'patient_examples'
    """
    best = prediction_results.get("best", {})
    model_name = best.get("model")
    feat_type = best.get("feature_type")
    if not model_name or not feat_type:
        raise ValueError("prediction_results['best'] is missing model/feature_type")

    model = prediction_results["models"].get((model_name, feat_type))
    if model is None:
        raise ValueError(f"Could not find trained model: {model_name} + {feat_type}")

    splits = prediction_results["splits"].get(feat_type)
    if splits is None:
        raise ValueError(f"No splits found for feature_type: {feat_type}")

    # Map back to feature_sets entry for feature names
    FEATURE_KEY_MAP = {
        "tfidf": "tfidf",
        "topic_distribution": "topic_lda",
        "structured": "structured",
        "combined": "combined",
        "text_stats": "text_stats",
        "embeddings": "embeddings",
    }
    feat_key = FEATURE_KEY_MAP.get(feat_type, feat_type)
    entry = feature_sets.get(feat_key, {})
    feature_names = entry.get("names")
    if feature_names is None:
        n_feat = splits["X_test"].shape[1]
        feature_names = [f"f{i}" for i in range(n_feat)]

    logger.info("Running SHAP analysis for %s + %s ...", model_name, feat_type)

    shap_out = compute_shap_values(
        model,
        X_explain=splits["X_test"],
        X_background=splits["X_train"],
        model_name=model_name,
        max_samples=n_explain,
    )

    global_imp = shap_global_importance(
        shap_out["shap_values"], feature_names, top_n=20,
    )
    logger.info("Top SHAP features:\n%s", global_imp.head(10).to_string(index=False))

    # Pick a few patient examples (highest predicted risk)
    patient_examples = []
    n_samples = shap_out["X"].shape[0]
    if n_samples > 0:
        # Patients with largest positive SHAP sum = highest predicted risk shift
        risk_scores = shap_out["shap_values"].sum(axis=1)
        top_idx = np.argsort(-risk_scores)[:n_patients_to_show]
        for idx in top_idx:
            patient_examples.append(explain_patient(
                shap_out["shap_values"],
                shap_out["X"],
                feature_names,
                patient_idx=int(idx),
                base_value=shap_out["base_value"],
                top_n=10,
            ))

    return {
        "model_name": model_name,
        "feature_type": feat_type,
        "shap_values": shap_out["shap_values"],
        "base_value": shap_out["base_value"],
        "X_test": shap_out["X"],
        "feature_names": feature_names,
        "global_importance": global_imp,
        "patient_examples": patient_examples,
    }
