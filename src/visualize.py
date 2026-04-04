"""
visualize.py
Purpose: Generate publication-quality figures for the Clinical NLP pipeline.

Plot types:
  1. Dataset distributions (demographics, readmission rates)
  2. Topic modeling (word clouds, coherence, topic-readmission heatmap)
  3. Prediction performance (ROC, PR curves, confusion matrices, feature importance)
  4. Fairness (disparity bar charts, group metric comparison)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DPI = 300
DEFAULT_FORMAT = "png"
DEFAULT_PALETTE = "Set2"
DEFAULT_SAVE_PATH = "results/figures"


def _setup_style(config: Optional[dict] = None):
    """Apply consistent plot styling."""
    cfg = (config or {}).get("visualization", {})
    palette = cfg.get("color_palette", DEFAULT_PALETTE)
    sns.set_theme(style="whitegrid", palette=palette, font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": cfg.get("figure_dpi", DEFAULT_DPI),
        "savefig.dpi": cfg.get("figure_dpi", DEFAULT_DPI),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


def _save_fig(fig, name: str, config: Optional[dict] = None):
    """Save figure to the configured output directory."""
    cfg = (config or {}).get("visualization", {})
    save_path = Path(cfg.get("save_path", DEFAULT_SAVE_PATH))
    fmt = cfg.get("figure_format", DEFAULT_FORMAT)
    save_path.mkdir(parents=True, exist_ok=True)
    filepath = save_path / f"{name}.{fmt}"
    fig.savefig(filepath)
    plt.close(fig)
    logger.info("Saved figure: %s", filepath)
    return str(filepath)


# =========================================================================
# 1. Dataset Distribution Plots
# =========================================================================

def plot_demographics(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> List[str]:
    """
    Plot demographic distributions: gender, age, insurance, readmission rate.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset from data_loader.

    Returns
    -------
    list of str — saved file paths
    """
    _setup_style(config)
    saved = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Demographics", fontsize=16, fontweight="bold")

    # Gender distribution
    if "gender" in df.columns:
        gender_counts = df["gender"].value_counts()
        axes[0, 0].bar(gender_counts.index, gender_counts.values, color=sns.color_palette()[:len(gender_counts)])
        axes[0, 0].set_title("Gender Distribution")
        axes[0, 0].set_ylabel("Count")
        for i, (idx, val) in enumerate(gender_counts.items()):
            axes[0, 0].text(i, val + 2, str(val), ha="center", fontweight="bold")

    # Age distribution
    age_col = "anchor_age" if "anchor_age" in df.columns else None
    if age_col:
        axes[0, 1].hist(df[age_col].dropna(), bins=20, edgecolor="black", alpha=0.7, color=sns.color_palette()[2])
        axes[0, 1].set_title("Age Distribution")
        axes[0, 1].set_xlabel("Age")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].axvline(df[age_col].median(), color="red", linestyle="--", label=f"Median: {df[age_col].median():.0f}")
        axes[0, 1].legend()

    # Insurance distribution
    if "insurance" in df.columns:
        ins_counts = df["insurance"].value_counts()
        axes[1, 0].bar(ins_counts.index, ins_counts.values, color=sns.color_palette()[:len(ins_counts)])
        axes[1, 0].set_title("Insurance Distribution")
        axes[1, 0].set_ylabel("Count")
        for i, (idx, val) in enumerate(ins_counts.items()):
            axes[1, 0].text(i, val + 2, str(val), ha="center", fontweight="bold")

    # Readmission rate by age group
    if "age_group" in df.columns and "readmission_30day" in df.columns:
        eligible = df[df["readmission_30day"] >= 0].copy()
        rates = eligible.groupby("age_group")["readmission_30day"].mean() * 100
        rates = rates.reindex(["<40", "40-65", "65+"])
        axes[1, 1].bar(rates.index, rates.values, color=sns.color_palette()[3])
        axes[1, 1].set_title("30-Day Readmission Rate by Age Group")
        axes[1, 1].set_ylabel("Readmission Rate (%)")
        for i, (idx, val) in enumerate(rates.items()):
            axes[1, 1].text(i, val + 0.3, f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    saved.append(_save_fig(fig, "demographics", config))

    return saved


def plot_note_length_distribution(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> str:
    """Plot distribution of note lengths (word count)."""
    _setup_style(config)

    word_counts = df["text"].dropna().apply(lambda t: len(str(t).split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Clinical Note Length Distribution", fontsize=14, fontweight="bold")

    axes[0].hist(word_counts, bins=30, edgecolor="black", alpha=0.7, color=sns.color_palette()[0])
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Number of Notes")
    axes[0].set_title("Histogram")
    axes[0].axvline(word_counts.median(), color="red", linestyle="--", label=f"Median: {word_counts.median():.0f}")
    axes[0].legend()

    # By readmission status
    if "readmission_30day" in df.columns:
        eligible = df[df["readmission_30day"] >= 0].copy()
        eligible["word_count"] = eligible["text"].apply(lambda t: len(str(t).split()))
        for label, group in eligible.groupby("readmission_30day"):
            tag = "Readmitted" if label == 1 else "Not Readmitted"
            axes[1].hist(group["word_count"], bins=20, alpha=0.5, label=tag, edgecolor="black")
        axes[1].set_xlabel("Word Count")
        axes[1].set_ylabel("Count")
        axes[1].set_title("By Readmission Status")
        axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "note_length_distribution", config)


# =========================================================================
# 2. Topic Modeling Plots
# =========================================================================

def plot_coherence_scores(
    coherence_scores: Dict[int, float],
    best_num_topics: int,
    config: Optional[dict] = None,
) -> str:
    """Plot coherence scores across topic counts with the optimal marked."""
    _setup_style(config)

    topics = sorted(coherence_scores.keys())
    scores = [coherence_scores[t] for t in topics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(topics, scores, "o-", linewidth=2, markersize=8, color=sns.color_palette()[0])
    ax.axvline(best_num_topics, color="red", linestyle="--", alpha=0.7, label=f"Best: {best_num_topics} topics")
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel("Coherence Score (c_v)")
    ax.set_title("LDA Topic Coherence vs. Number of Topics")
    ax.set_xticks(topics)
    ax.legend()
    plt.tight_layout()
    return _save_fig(fig, "coherence_scores", config)


def plot_topic_word_clouds(
    topic_words: Dict[int, List[Tuple[str, float]]],
    topic_labels: Dict[int, str],
    config: Optional[dict] = None,
) -> str:
    """Generate word clouds for each topic."""
    _setup_style(config)

    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud not installed — skipping word cloud plot.")
        return ""

    n_topics = len(topic_words)
    cols = min(3, n_topics)
    rows = (n_topics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_topics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (tid, words) in enumerate(topic_words.items()):
        word_freq = {w: wt for w, wt in words}
        wc = WordCloud(
            width=400, height=300,
            background_color="white",
            colormap="viridis",
            max_words=30,
        ).generate_from_frequencies(word_freq)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Topic {tid}: {topic_labels.get(tid, '')}", fontsize=11)
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Topic Word Clouds (LDA)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_fig(fig, "topic_word_clouds", config)


def plot_topic_readmission_heatmap(
    doc_topic_matrix: np.ndarray,
    readmission_labels: np.ndarray,
    topic_labels: Dict[int, str],
    config: Optional[dict] = None,
) -> str:
    """Heatmap of mean topic prevalence by readmission status."""
    _setup_style(config)

    labels = np.array(readmission_labels)
    data = {}
    for status, name in [(0, "Not Readmitted"), (1, "Readmitted")]:
        mask = labels == status
        if mask.sum() > 0:
            data[name] = doc_topic_matrix[mask].mean(axis=0)

    heatmap_df = pd.DataFrame(data, index=[topic_labels.get(i, f"T{i}") for i in range(doc_topic_matrix.shape[1])])

    fig, ax = plt.subplots(figsize=(8, max(4, len(heatmap_df) * 0.5)))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Mean Topic Prevalence by Readmission Status", fontsize=13, fontweight="bold")
    ax.set_ylabel("Topic")
    plt.tight_layout()
    return _save_fig(fig, "topic_readmission_heatmap", config)


# =========================================================================
# 3. Prediction Performance Plots
# =========================================================================

def plot_roc_curves(
    results: List[dict],
    config: Optional[dict] = None,
) -> str:
    """Plot ROC curves for all model-feature combinations."""
    _setup_style(config)

    # Group by feature type
    feature_types = sorted(set(r.get("feature_type", "unknown") for r in results))
    n_panels = len(feature_types)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = sns.color_palette("husl", n_colors=10)

    for ax, feat_type in zip(axes, feature_types):
        subset = [r for r in results if r.get("feature_type") == feat_type]
        for i, r in enumerate(subset):
            curve = r.get("roc_curve", {})
            fpr, tpr = curve.get("fpr", []), curve.get("tpr", [])
            if len(fpr) > 0:
                ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})", color=colors[i % len(colors)], linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {feat_type}")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("ROC Curves", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "roc_curves", config)


def plot_pr_curves(
    results: List[dict],
    config: Optional[dict] = None,
) -> str:
    """Plot Precision-Recall curves for all model-feature combinations."""
    _setup_style(config)

    feature_types = sorted(set(r.get("feature_type", "unknown") for r in results))
    n_panels = len(feature_types)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = sns.color_palette("husl", n_colors=10)

    for ax, feat_type in zip(axes, feature_types):
        subset = [r for r in results if r.get("feature_type") == feat_type]
        for i, r in enumerate(subset):
            curve = r.get("pr_curve", {})
            prec, rec = curve.get("precision", []), curve.get("recall", [])
            if len(prec) > 0:
                ax.plot(rec, prec, label=f"{r['model']} (AP={r['pr_auc']:.3f})", color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR — {feat_type}")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "pr_curves", config)


def plot_confusion_matrices(
    results: List[dict],
    config: Optional[dict] = None,
) -> str:
    """Plot confusion matrices for each model result."""
    _setup_style(config)

    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for i, r in enumerate(results):
        cm = np.array(r.get("confusion_matrix", [[0, 0], [0, 0]]))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=axes[i],
        )
        feat_type = r.get("feature_type", "")
        axes[i].set_title(f"{r['model']}\n({feat_type})", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "confusion_matrices", config)


def plot_model_comparison(
    results_df: pd.DataFrame,
    config: Optional[dict] = None,
) -> str:
    """Bar chart comparing model performance across feature types."""
    _setup_style(config)

    metrics = ["roc_auc", "f1", "precision", "recall"]
    available = [m for m in metrics if m in results_df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        pivot = results_df.pivot_table(index="model", columns="feature_type", values=metric)
        pivot.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85)
        ax.set_title(metric.upper().replace("_", "-"), fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.legend(fontsize=7, title="Features")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "model_comparison", config)


def plot_feature_importance(
    importances: Dict[tuple, pd.DataFrame],
    top_n: int = 15,
    config: Optional[dict] = None,
) -> str:
    """Plot top feature importances for each model-feature combination."""
    _setup_style(config)

    n = len(importances)
    if n == 0:
        logger.info("No feature importances to plot.")
        return ""

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for i, ((model_name, feat_type), imp_df) in enumerate(importances.items()):
        top = imp_df.head(top_n).sort_values("abs_importance")
        axes[i].barh(top["feature"], top["abs_importance"], color=sns.color_palette()[i % 8])
        axes[i].set_title(f"{model_name} — {feat_type}", fontsize=10)
        axes[i].set_xlabel("Importance")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "feature_importance", config)


# =========================================================================
# 4. Fairness Plots
# =========================================================================

def plot_fairness_disparity(
    summary_df: pd.DataFrame,
    config: Optional[dict] = None,
) -> str:
    """Bar chart of fairness disparity metrics across protected attributes."""
    _setup_style(config)

    if summary_df.empty:
        logger.info("No fairness summary to plot.")
        return ""

    metrics = [c for c in summary_df.columns if c != "attribute"]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", n_colors=len(summary_df))

    for ax, metric in zip(axes, metrics):
        values = summary_df[metric].abs()
        bars = ax.bar(summary_df.index, values, color=colors[:len(values)], edgecolor="black")
        ax.axhline(0.1, color="red", linestyle="--", alpha=0.7, label="Threshold (0.1)")
        ax.set_title(metric.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_ylabel("Absolute Disparity")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=9)

    fig.suptitle("Fairness Disparity Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "fairness_disparity", config)


def plot_fairness_group_metrics(
    group_metrics: Dict[str, pd.DataFrame],
    config: Optional[dict] = None,
) -> str:
    """Grouped bar chart of per-group metrics for each protected attribute."""
    _setup_style(config)

    n_attrs = len(group_metrics)
    if n_attrs == 0:
        return ""

    plot_metrics = ["accuracy", "precision", "recall", "f1", "fpr"]

    fig, axes = plt.subplots(1, n_attrs, figsize=(7 * n_attrs, 5))
    if n_attrs == 1:
        axes = [axes]

    for ax, (attr, df) in zip(axes, group_metrics.items()):
        available = [m for m in plot_metrics if m in df.columns]
        df[available].plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85)
        ax.set_title(f"Metrics by {attr.title()}", fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_xlabel(attr.title())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Per-Group Classification Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "fairness_group_metrics", config)


# =========================================================================
# 5. Generate all figures
# =========================================================================

def generate_all_figures(
    merged_df: pd.DataFrame,
    prediction_results: Optional[dict] = None,
    lda_results: Optional[dict] = None,
    fairness_results: Optional[dict] = None,
    config: Optional[dict] = None,
) -> List[str]:
    """
    Generate all available figures based on the results provided.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged dataset from data_loader.
    prediction_results : dict, optional
        Output of predict.run_prediction_pipeline().
    lda_results : dict, optional
        Output of topic_model.run_lda_pipeline().
    fairness_results : dict, optional
        Output of fairness.run_fairness_audit().
    config : dict, optional
        Loaded config.yaml.

    Returns
    -------
    list of str — paths to all saved figures
    """
    saved = []

    # Dataset distributions
    logger.info("Generating dataset distribution plots ...")
    saved.extend(plot_demographics(merged_df, config))
    saved.append(plot_note_length_distribution(merged_df, config))

    # Topic modeling
    if lda_results is not None:
        logger.info("Generating topic modeling plots ...")
        saved.append(plot_coherence_scores(
            lda_results["coherence_scores"],
            lda_results["best_num_topics"],
            config,
        ))
        saved.append(plot_topic_word_clouds(
            lda_results["topic_words"],
            lda_results["topic_labels"],
            config,
        ))
        if "doc_topic_matrix" in lda_results and "readmission_labels" in lda_results:
            saved.append(plot_topic_readmission_heatmap(
                lda_results["doc_topic_matrix"],
                lda_results["readmission_labels"],
                lda_results["topic_labels"],
                config,
            ))

    # Prediction
    if prediction_results is not None:
        logger.info("Generating prediction performance plots ...")
        results = prediction_results.get("results", [])
        if results:
            saved.append(plot_roc_curves(results, config))
            saved.append(plot_pr_curves(results, config))
            saved.append(plot_confusion_matrices(results, config))

        results_df = prediction_results.get("results_df")
        if results_df is not None and not results_df.empty:
            saved.append(plot_model_comparison(results_df, config))

        importances = prediction_results.get("importances", {})
        if importances:
            saved.append(plot_feature_importance(importances, config=config))

    # Fairness
    if fairness_results is not None:
        logger.info("Generating fairness plots ...")
        summary_df = fairness_results.get("summary_df")
        if summary_df is not None:
            saved.append(plot_fairness_disparity(summary_df, config))

        group_metrics = fairness_results.get("group_metrics", {})
        if group_metrics:
            saved.append(plot_fairness_group_metrics(group_metrics, config))

    saved = [s for s in saved if s]  # remove empty strings
    logger.info("Generated %d figures total.", len(saved))
    return saved
