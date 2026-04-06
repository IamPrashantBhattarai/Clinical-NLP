"""
predict.py
Purpose: Train and evaluate readmission prediction models using features from feature_engineer.py.

Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Handles class imbalance via SMOTE and class weights.
Reports: accuracy, precision, recall, F1, ROC-AUC, PR-AUC per feature set.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Train / val / test split
# ---------------------------------------------------------------------------

def split_data(
    X,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Stratified split into train / validation / test sets.

    Parameters
    ----------
    X : sparse matrix or np.ndarray
    y : np.ndarray of binary labels
    test_size : float
        Fraction held out for final test.
    val_size : float
        Fraction of the remaining data used for validation.
    random_seed : int

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed,
    )

    # val_size is relative to the full dataset; adjust for the remaining portion
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=random_seed,
    )

    logger.info(
        "Split — train: %d, val: %d, test: %d  (pos rate: train=%.1f%%, val=%.1f%%, test=%.1f%%)",
        len(y_train), len(y_val), len(y_test),
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# 2. SMOTE resampling (optional)
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train, random_seed: int = 42):
    """
    Apply SMOTE to oversample the minority class in the training set.

    Returns resampled X_train, y_train.  Skips if classes are already balanced
    (minority >= 40% of majority).
    """
    from imblearn.over_sampling import SMOTE

    counts = np.bincount(y_train)
    minority_ratio = counts.min() / counts.max()
    if minority_ratio >= 0.4:
        logger.info("Classes already near-balanced (ratio=%.2f) — skipping SMOTE.", minority_ratio)
        return X_train, y_train

    sm = SMOTE(random_state=random_seed)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(
        "SMOTE — before: %s, after: %s",
        dict(zip(*np.unique(y_train, return_counts=True))),
        dict(zip(*np.unique(y_res, return_counts=True))),
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# 3. Model factory
# ---------------------------------------------------------------------------

def get_model(name: str, random_seed: int = 42):
    """
    Return an untrained sklearn-compatible estimator.

    Supported names: logistic_regression, random_forest, xgboost, lightgbm.
    """
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            penalty="l2",
            C=1.0,
            random_state=random_seed,
            n_jobs=1,
        )

    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=1,
        )

    elif name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # will be overridden in train_model
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=random_seed,
            n_jobs=1,
        )

    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=random_seed,
            n_jobs=1,
            verbose=-1,
        )

    else:
        raise ValueError(f"Unknown model name: '{name}'")


# ---------------------------------------------------------------------------
# 4. Train a single model
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    use_smote: bool = False,
    random_seed: int = 42,
):
    """
    Train a model with optional SMOTE and early stopping (for boosters).

    Returns the fitted model.
    """
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_seed=random_seed)

    model = get_model(model_name, random_seed=random_seed)

    # Set scale_pos_weight for XGBoost
    if model_name == "xgboost":
        neg, pos = np.bincount(y_train)
        model.set_params(scale_pos_weight=neg / pos if pos > 0 else 1.0)

    # Early stopping for boosted models
    if model_name in ("xgboost", "lightgbm") and X_val is not None:
        fit_params = {
            "eval_set": [(X_val, y_val)],
        }
        if model_name == "xgboost":
            fit_params["verbose"] = False
        else:
            fit_params["callbacks"] = [
                _lgbm_early_stopping(50),
                _lgbm_log_eval(-1),
            ]
        model.fit(X_train, y_train, **fit_params)
    else:
        model.fit(X_train, y_train)

    logger.info("Trained %s.", model_name)
    return model


def _lgbm_early_stopping(stopping_rounds):
    """LightGBM early stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=stopping_rounds, verbose=False)


def _lgbm_log_eval(period):
    """LightGBM log evaluation callback."""
    from lightgbm import log_evaluation
    return log_evaluation(period=period)


# ---------------------------------------------------------------------------
# 5. Evaluate a single model
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "",
    threshold: float = 0.5,
) -> dict:
    """
    Compute standard classification metrics on the test set.

    Returns a dict of metrics + arrays for curve plotting.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Guard against single-class test sets (can happen with small data + stratified splits)
    n_classes = len(np.unique(y_test))
    if n_classes < 2:
        logger.warning("%s — only one class in y_test, ROC-AUC/PR-AUC undefined.", model_name)
        roc_auc_val = 0.0
        pr_auc_val = 0.0
    else:
        roc_auc_val = round(roc_auc_score(y_test, y_prob), 4)
        pr_auc_val = round(average_precision_score(y_test, y_prob), 4)

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Curve data for plotting
    if n_classes >= 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        prec_curve, rec_curve = np.array([0, 1]), np.array([1, 0])

    metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
    metrics["pr_curve"] = {"precision": prec_curve, "recall": rec_curve}
    metrics["y_prob"] = y_prob

    logger.info(
        "%s — Acc: %.3f | F1: %.3f | ROC-AUC: %.3f | PR-AUC: %.3f",
        model_name, metrics["accuracy"], metrics["f1"],
        metrics["roc_auc"], metrics["pr_auc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# 6. Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model_name: str,
    X,
    y: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> dict:
    """
    Stratified k-fold cross-validation. Returns mean and std of key metrics.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        if issparse(X):
            X_tr, X_te = X[train_idx], X[test_idx]
        else:
            X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = get_model(model_name, random_seed=random_seed)
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_metrics.append({
            "fold": fold,
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_te, y_prob),
            "pr_auc": average_precision_score(y_te, y_prob),
        })

    df = pd.DataFrame(fold_metrics)
    summary = {
        "model": model_name,
        "n_folds": n_folds,
        "f1_mean": round(df["f1"].mean(), 4),
        "f1_std": round(df["f1"].std(), 4),
        "roc_auc_mean": round(df["roc_auc"].mean(), 4),
        "roc_auc_std": round(df["roc_auc"].std(), 4),
        "pr_auc_mean": round(df["pr_auc"].mean(), 4),
        "pr_auc_std": round(df["pr_auc"].std(), 4),
        "fold_details": fold_metrics,
    }
    logger.info(
        "CV %s (%d folds) — F1: %.3f±%.3f | ROC-AUC: %.3f±%.3f",
        model_name, n_folds, summary["f1_mean"], summary["f1_std"],
        summary["roc_auc_mean"], summary["roc_auc_std"],
    )
    return summary


# ---------------------------------------------------------------------------
# 7. Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model,
    feature_names: List[str],
    model_name: str = "",
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Extract feature importances from a fitted model.

    Returns a DataFrame sorted by absolute importance, top_n rows.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_.ravel()
    else:
        logger.warning("Model %s has no feature_importances_ or coef_.", model_name)
        return pd.DataFrame()

    # Align lengths (feature_names may be longer if combined includes sparse names)
    n = min(len(importances), len(feature_names))
    df = pd.DataFrame({
        "feature": feature_names[:n],
        "importance": importances[:n],
        "abs_importance": np.abs(importances[:n]),
    })
    df = df.sort_values("abs_importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 8. Find optimal threshold
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Search thresholds from 0.1 to 0.9 to maximize the chosen metric.

    Parameters
    ----------
    metric : str
        'f1' or 'youden' (Youden's J = sensitivity + specificity - 1).

    Returns
    -------
    tuple
        (best_threshold, best_score)
    """
    best_thresh, best_score = 0.5, 0.0
    for thresh in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = thresh

    logger.info("Optimal threshold (metric=%s): %.2f (score=%.4f)", metric, best_thresh, best_score)
    return round(best_thresh, 2), round(best_score, 4)


# ---------------------------------------------------------------------------
# 9. Full prediction pipeline
# ---------------------------------------------------------------------------

def run_prediction_pipeline(
    feature_sets: dict,
    config: Optional[dict] = None,
) -> dict:
    """
    Train all configured models on each feature type. Evaluate on test set.

    Parameters
    ----------
    feature_sets : dict
        Output of feature_engineer.build_feature_sets(). Must contain 'label'
        and feature-set entries like 'tfidf', 'structured', 'combined', etc.
    config : dict, optional
        Loaded config.yaml.

    Returns
    -------
    dict with keys:
        'results'        — list of per-model-per-feature-set metric dicts
        'models'         — {(model_name, feature_type): fitted_model}
        'splits'         — {feature_type: split_dict}
        'best'           — dict describing the best model/feature combo
        'cv_results'     — list of cross-validation summaries
        'importances'    — {(model_name, feature_type): DataFrame}
        'results_df'     — summary DataFrame
    """
    cfg = config or {}
    pred_cfg = cfg.get("prediction", {})
    model_names = pred_cfg.get("models", [
        "logistic_regression", "random_forest", "xgboost", "lightgbm",
    ])
    feature_types = pred_cfg.get("feature_types", [
        "tfidf", "topic_distribution", "structured", "combined",
    ])
    test_size = pred_cfg.get("test_size", 0.2)
    val_size = pred_cfg.get("val_size", 0.1)
    random_seed = cfg.get("data", {}).get("random_seed", 42)

    labels = feature_sets.get("label")
    if labels is None:
        raise ValueError("feature_sets must contain 'label' array.")

    # Map config feature type names to feature_sets keys
    FEATURE_KEY_MAP = {
        "tfidf": "tfidf",
        "topic_distribution": "topic_lda",
        "structured": "structured",
        "combined": "combined",
        "text_stats": "text_stats",
    }

    all_results = []
    all_models = {}
    all_splits = {}
    all_cv = []
    all_importances = {}

    for feat_type in feature_types:
        feat_key = FEATURE_KEY_MAP.get(feat_type, feat_type)
        entry = feature_sets.get(feat_key)
        if entry is None:
            logger.warning("Feature set '%s' not available — skipping.", feat_type)
            continue

        X = entry["X"]
        feat_names = entry.get("names", [f"f{i}" for i in range(X.shape[1])])

        logger.info("=" * 60)
        logger.info("Feature set: %s — shape %s", feat_type, X.shape)
        logger.info("=" * 60)

        # Split
        splits = split_data(X, labels, test_size=test_size, val_size=val_size, random_seed=random_seed)
        all_splits[feat_type] = splits

        for model_name in model_names:
            logger.info("--- %s + %s ---", model_name, feat_type)

            # Train
            model = train_model(
                model_name,
                splits["X_train"], splits["y_train"],
                X_val=splits["X_val"], y_val=splits["y_val"],
                use_smote=False,
                random_seed=random_seed,
            )
            all_models[(model_name, feat_type)] = model

            # Evaluate on test set
            metrics = evaluate_model(
                model, splits["X_test"], splits["y_test"], model_name=model_name,
            )
            metrics["feature_type"] = feat_type

            # Optimal threshold
            opt_thresh, opt_f1 = find_optimal_threshold(
                splits["y_test"], metrics["y_prob"], metric="f1",
            )
            metrics["optimal_threshold"] = opt_thresh
            metrics["f1_at_optimal"] = opt_f1
            all_results.append(metrics)

            # Feature importance
            imp_df = get_feature_importance(model, feat_names, model_name=model_name)
            if not imp_df.empty:
                all_importances[(model_name, feat_type)] = imp_df

    # Cross-validate best model per feature type on combined split
    logger.info("=" * 60)
    logger.info("Cross-validation on best model per feature type")
    logger.info("=" * 60)

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("roc_curve", "pr_curve", "y_prob", "confusion_matrix")}
        for r in all_results
    ])

    if not results_df.empty:
        for feat_type in results_df["feature_type"].unique():
            subset = results_df[results_df["feature_type"] == feat_type]
            best_row = subset.loc[subset["roc_auc"].idxmax()]
            best_model_name = best_row["model"]
            feat_key = FEATURE_KEY_MAP.get(feat_type, feat_type)
            entry = feature_sets.get(feat_key)
            if entry is not None:
                cv_result = cross_validate_model(
                    best_model_name, entry["X"], labels,
                    n_folds=5, random_seed=random_seed,
                )
                cv_result["feature_type"] = feat_type
                all_cv.append(cv_result)

    # Identify overall best
    best = {}
    if not results_df.empty:
        best_idx = results_df["roc_auc"].idxmax()
        best = results_df.iloc[best_idx].to_dict()
        logger.info(
            "Best: %s + %s — ROC-AUC: %.4f, F1: %.4f",
            best.get("model"), best.get("feature_type"),
            best.get("roc_auc", 0), best.get("f1", 0),
        )

    # Print summary table
    if not results_df.empty:
        display_cols = ["model", "feature_type", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        logger.info("\n%s", results_df[display_cols].to_string(index=False))

    return {
        "results": all_results,
        "models": all_models,
        "splits": all_splits,
        "best": best,
        "cv_results": all_cv,
        "importances": all_importances,
        "results_df": results_df,
    }
