"""
fairness.py
Purpose: Evaluate prediction fairness across protected attributes (gender, insurance, age_group).

Uses Fairlearn to compute:
  - Demographic parity difference
  - Equalized odds difference
  - False positive rate difference
  - False negative rate difference

Also provides per-group metric breakdowns and a summary report.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    false_negative_rate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Per-group metric breakdown
# ---------------------------------------------------------------------------

def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive_features: pd.Series,
    attribute_name: str = "",
) -> pd.DataFrame:
    """
    Compute classification metrics for each group within a protected attribute.

    Parameters
    ----------
    y_true : array-like of true labels
    y_pred : array-like of predicted labels
    y_prob : array-like of predicted probabilities
    sensitive_features : pd.Series with group labels (e.g., 'M'/'F')
    attribute_name : str for labeling output

    Returns
    -------
    pd.DataFrame with one row per group and columns for each metric.
    """
    metric_fns = {
        "accuracy": accuracy_score,
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall": lambda y, p: recall_score(y, p, zero_division=0),
        "f1": lambda y, p: f1_score(y, p, zero_division=0),
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "selection_rate": lambda y, p: float(np.asarray(p).mean()),
    }

    mf = MetricFrame(
        metrics=metric_fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    df = mf.by_group.copy()
    df.index.name = attribute_name or "group"
    df = df.round(4)

    # Add group sizes and positive rates
    groups = pd.Series(sensitive_features)
    group_sizes = groups.value_counts()
    positive_rates = pd.Series(y_true).groupby(groups.values).mean()

    df.insert(0, "n_samples", group_sizes)
    df.insert(1, "positive_rate", positive_rates.round(4))

    logger.info("Group metrics for '%s':\n%s", attribute_name, df.to_string())
    return df


# ---------------------------------------------------------------------------
# 2. Fairness disparity metrics
# ---------------------------------------------------------------------------

def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    attribute_name: str = "",
) -> dict:
    """
    Compute aggregate fairness disparity metrics.

    Returns
    -------
    dict with keys: attribute, demographic_parity_difference,
    equalized_odds_difference, fpr_difference, fnr_difference
    """
    dpd = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features,
    )
    eod = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features,
    )

    # FPR and FNR differences (max - min across groups)
    fpr_frame = MetricFrame(
        metrics=false_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    fnr_frame = MetricFrame(
        metrics=false_negative_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    result = {
        "attribute": attribute_name,
        "demographic_parity_difference": round(dpd, 4),
        "equalized_odds_difference": round(eod, 4),
        "fpr_difference": round(fpr_frame.difference(), 4),
        "fnr_difference": round(fnr_frame.difference(), 4),
    }

    logger.info(
        "Fairness [%s] — DPD: %.4f | EOD: %.4f | FPR diff: %.4f | FNR diff: %.4f",
        attribute_name, result["demographic_parity_difference"],
        result["equalized_odds_difference"],
        result["fpr_difference"], result["fnr_difference"],
    )
    return result


# ---------------------------------------------------------------------------
# 3. Full fairness audit
# ---------------------------------------------------------------------------

def run_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    protected_df: pd.DataFrame,
    config: Optional[dict] = None,
) -> dict:
    """
    Run a complete fairness audit across all configured protected attributes.

    Parameters
    ----------
    y_true : array-like of true labels
    y_pred : array-like of predicted labels
    y_prob : array-like of predicted probabilities
    protected_df : DataFrame with columns for each protected attribute
        (e.g., 'gender', 'insurance', 'age_group'). Must be aligned with y_true.
    config : dict, optional
        Loaded config.yaml. Uses fairness.protected_attributes if provided.

    Returns
    -------
    dict with keys:
        'group_metrics'    — {attribute: DataFrame}
        'fairness_metrics' — {attribute: dict}
        'summary_df'       — DataFrame summarizing disparity across attributes
    """
    cfg = config or {}
    fairness_cfg = cfg.get("fairness", {})
    attributes = fairness_cfg.get(
        "protected_attributes",
        [col for col in protected_df.columns],
    )

    group_metrics = {}
    fairness_metrics = {}

    for attr in attributes:
        if attr not in protected_df.columns:
            logger.warning("Attribute '%s' not found in protected_df — skipping.", attr)
            continue

        sensitive = protected_df[attr].copy()

        # Drop rows with missing attribute values
        mask = sensitive.notna()
        if mask.sum() < len(mask):
            logger.info("Dropping %d rows with missing '%s'.", (~mask).sum(), attr)

        y_t = np.asarray(y_true)[mask]
        y_p = np.asarray(y_pred)[mask]
        y_pr = np.asarray(y_prob)[mask]
        sf = sensitive[mask].reset_index(drop=True)

        group_metrics[attr] = compute_group_metrics(y_t, y_p, y_pr, sf, attribute_name=attr)
        fairness_metrics[attr] = compute_fairness_metrics(y_t, y_p, sf, attribute_name=attr)

    # Summary table
    summary_df = pd.DataFrame(list(fairness_metrics.values()))
    if not summary_df.empty:
        summary_df = summary_df.set_index("attribute")

    _print_fairness_report(group_metrics, summary_df)

    return {
        "group_metrics": group_metrics,
        "fairness_metrics": fairness_metrics,
        "summary_df": summary_df,
    }


# ---------------------------------------------------------------------------
# 4. Report printer
# ---------------------------------------------------------------------------

def _print_fairness_report(
    group_metrics: dict,
    summary_df: pd.DataFrame,
) -> None:
    """Print a formatted fairness audit report."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("FAIRNESS AUDIT REPORT")
    print(sep)

    for attr, df in group_metrics.items():
        print(f"\n--- {attr.upper()} ---")
        print(df.to_string())

    print(f"\n--- DISPARITY SUMMARY ---")
    if not summary_df.empty:
        print(summary_df.to_string())

        # Flag concerning disparities (threshold: 0.1)
        threshold = 0.1
        flags = []
        for col in summary_df.columns:
            for attr in summary_df.index:
                val = abs(summary_df.loc[attr, col])
                if val > threshold:
                    flags.append(f"  WARNING: {attr} — {col} = {val:.4f} (> {threshold})")

        if flags:
            print(f"\n--- FAIRNESS FLAGS ---")
            for f in flags:
                print(f)
        else:
            print("\n  All disparity metrics within acceptable range (< 0.1).")
    else:
        print("  No fairness metrics computed.")

    print(sep)
