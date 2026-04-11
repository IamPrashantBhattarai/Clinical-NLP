"""
exporter.py
Serialize pipeline outputs into the JSON files the dashboard reads.

The FastAPI backend looks in results/exports/ for three files:
  - results.json   (model comparison metrics)
  - fairness.json  (group-level fairness audit)
  - topics.json    (LDA topics + readmission association)

If any file is missing, the dashboard falls back to mock data. Call
export_dashboard_json() from the notebook (or a script) after running the
relevant pipeline stages.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_DIR = Path("results/exports")

# Fields the dashboard expects in every row of results.json["results"]
RESULT_FIELDS = ("model", "feature_type", "accuracy", "precision",
                 "recall", "f1", "roc_auc", "pr_auc")

# Fields the dashboard expects on every group row in fairness.json
GROUP_FIELDS = ("accuracy", "precision", "recall", "f1", "fpr", "selection_rate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(v: Any, default: float = 0.0) -> float:
    """Coerce any numeric-ish value to a plain float, NaN-safe."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s (%d bytes)", path, path.stat().st_size)


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------

def build_results_payload(prediction_results: Dict) -> Dict:
    """Convert run_prediction_pipeline() output to the dashboard's shape."""
    raw = prediction_results.get("results") or []
    rows: List[Dict] = []
    for r in raw:
        rows.append({
            "model":        str(r.get("model", "unknown")),
            "feature_type": str(r.get("feature_type", "unknown")),
            "accuracy":     _to_float(r.get("accuracy")),
            "precision":    _to_float(r.get("precision")),
            "recall":       _to_float(r.get("recall")),
            "f1":           _to_float(r.get("f1")),
            "roc_auc":      _to_float(r.get("roc_auc")),
            "pr_auc":       _to_float(r.get("pr_auc")),
        })

    best_raw = prediction_results.get("best") or {}
    best = None
    if best_raw:
        best = {k: best_raw.get(k) for k in RESULT_FIELDS if k in best_raw}
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            if k in best:
                best[k] = _to_float(best[k])

    n_models = len({r["model"] for r in rows})
    n_feature_sets = len({r["feature_type"] for r in rows})

    return {
        "results": rows,
        "best": best,
        "n_models": n_models,
        "n_feature_sets": n_feature_sets,
    }


# ---------------------------------------------------------------------------
# Fairness export
# ---------------------------------------------------------------------------

def build_fairness_payload(fairness_results: Dict) -> Dict:
    """Convert run_fairness_audit() output to the dashboard's shape."""
    group_metrics: Dict[str, pd.DataFrame] = fairness_results.get("group_metrics") or {}
    disparities: Dict[str, Dict] = fairness_results.get("fairness_metrics") or {}

    attrs: List[Dict] = []
    for attr, df in group_metrics.items():
        groups = []
        for group_label, row in df.iterrows():
            groups.append({
                "group":          str(group_label),
                "accuracy":       _to_float(row.get("accuracy")),
                "precision":      _to_float(row.get("precision")),
                "recall":         _to_float(row.get("recall")),
                "f1":             _to_float(row.get("f1")),
                "fpr":            _to_float(row.get("fpr")),
                "selection_rate": _to_float(row.get("selection_rate")),
            })

        d = disparities.get(attr, {})
        attrs.append({
            "attribute": str(attr),
            "groups": groups,
            "demographic_parity_difference": _to_float(d.get("demographic_parity_difference")),
            "equalized_odds_difference":     _to_float(d.get("equalized_odds_difference")),
            "fpr_difference":                _to_float(d.get("fpr_difference")),
            "fnr_difference":                _to_float(d.get("fnr_difference")),
        })

    return {"attributes": attrs}


# ---------------------------------------------------------------------------
# Topics export
# ---------------------------------------------------------------------------

def build_topics_payload(
    lda_results: Dict,
    readmission_labels: Optional[np.ndarray] = None,
) -> Dict:
    """
    Convert run_lda_pipeline() output to the dashboard's shape.

    If readmission_labels is provided and aligned with the doc-topic matrix,
    each topic's `readmission_rate` is computed as the mean readmission label
    among documents whose dominant topic is that topic.
    """
    topic_words: Dict[int, List] = lda_results.get("topic_words") or {}
    topic_labels: Dict[int, str] = lda_results.get("topic_labels") or {}
    doc_topic_matrix = lda_results.get("doc_topic_matrix")
    best_num_topics = lda_results.get("best_num_topics") or len(topic_words)
    coherence_scores = lda_results.get("coherence_scores") or {}

    # Pick the coherence score for the chosen topic count
    coherence = None
    if isinstance(coherence_scores, dict) and coherence_scores:
        coherence = coherence_scores.get(best_num_topics)
        if coherence is None:
            coherence = max(coherence_scores.values())

    # Per-topic readmission rate via dominant-topic assignment
    per_topic_rate: Dict[int, float] = {}
    if (
        doc_topic_matrix is not None
        and readmission_labels is not None
        and len(readmission_labels) == len(doc_topic_matrix)
    ):
        dominant = np.argmax(doc_topic_matrix, axis=1)
        y = np.asarray(readmission_labels).astype(float)
        for tid in range(doc_topic_matrix.shape[1]):
            mask = dominant == tid
            if mask.any():
                per_topic_rate[int(tid)] = float(y[mask].mean())

    topics = []
    for tid, words in sorted(topic_words.items()):
        word_entries = []
        for w, weight in words[:12]:
            word_entries.append({"word": str(w), "weight": _to_float(weight)})
        topics.append({
            "topic_id":         int(tid),
            "label":            str(topic_labels.get(tid, f"Topic {tid}")),
            "words":            word_entries,
            "readmission_rate": per_topic_rate.get(int(tid)),
        })

    return {
        "topics": topics,
        "coherence_score": _to_float(coherence) if coherence is not None else None,
        "n_topics": len(topics),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def export_dashboard_json(
    prediction_results: Optional[Dict] = None,
    fairness_results: Optional[Dict] = None,
    lda_results: Optional[Dict] = None,
    readmission_labels: Optional[np.ndarray] = None,
    output_dir: Path = DEFAULT_EXPORT_DIR,
) -> Dict[str, Path]:
    """
    Write the three dashboard JSON files from in-memory pipeline outputs.

    Any argument can be omitted — the corresponding file just won't be written,
    and the dashboard will continue to serve its mock fallback for that section.

    Returns a dict mapping section name to the path that was written.
    """
    out_dir = Path(output_dir)
    written: Dict[str, Path] = {}

    if prediction_results is not None:
        path = out_dir / "results.json"
        _write_json(path, build_results_payload(prediction_results))
        written["results"] = path

    if fairness_results is not None:
        path = out_dir / "fairness.json"
        _write_json(path, build_fairness_payload(fairness_results))
        written["fairness"] = path

    if lda_results is not None:
        path = out_dir / "topics.json"
        _write_json(path, build_topics_payload(lda_results, readmission_labels))
        written["topics"] = path

    if not written:
        logger.warning("export_dashboard_json: no inputs provided, nothing written.")
    else:
        logger.info("Exported %d dashboard file(s) to %s", len(written), out_dir)
    return written
