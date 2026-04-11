"""
feature_engineer.py
Purpose: Create features for readmission prediction from clinical notes and structured data.

Feature sets:
  1. TF-IDF          — text bag-of-words representation
  2. Topic features  — LDA / BERTopic document-topic distributions
  3. Structured      — demographics, admission info, prior utilization
  4. Text statistics — note length, negations, section count, etc.
  5. Combined        — all of the above stacked together
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. TF-IDF features
# ---------------------------------------------------------------------------

def create_tfidf_features(
    texts: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
    vectorizer=None,
    fit: bool = True,
):
    """
    Create TF-IDF feature matrix from a list of text strings.

    Parameters
    ----------
    texts : list of str
        Pre-joined token strings (e.g. " ".join(tokens) for each document).
    max_features : int
        Vocabulary size cap.
    ngram_range : tuple
        (min_n, max_n) for n-gram extraction.
    min_df : int
        Minimum document frequency for a term to be kept.
    max_df : float
        Maximum document frequency (as fraction) for a term to be kept.
    vectorizer : fitted TfidfVectorizer or None
        If provided and fit=False, transforms using existing vectorizer.
    fit : bool
        If True, fits a new vectorizer on `texts`.
        If False, uses the provided `vectorizer` to transform only.

    Returns
    -------
    tuple
        (TfidfVectorizer, scipy sparse matrix of shape (n_docs, n_features))
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if fit:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,     # log(1 + tf) — helps with long clinical notes
        )
        matrix = vectorizer.fit_transform(texts)
        logger.info(
            "TF-IDF fitted — shape: %s, vocab: %d", matrix.shape, len(vectorizer.vocabulary_)
        )
    else:
        if vectorizer is None:
            raise ValueError("vectorizer must be provided when fit=False")
        matrix = vectorizer.transform(texts)
        logger.info("TF-IDF transformed — shape: %s", matrix.shape)

    return vectorizer, matrix


# ---------------------------------------------------------------------------
# 2. Topic distribution features
# ---------------------------------------------------------------------------

def create_topic_features(
    model,
    corpus_or_texts,
    model_type: str = "lda",
) -> np.ndarray:
    """
    Extract topic probability distributions per document.

    Parameters
    ----------
    model : trained LDA model or BERTopic model
    corpus_or_texts : list
        Gensim BoW corpus for LDA, or list of strings for BERTopic.
    model_type : str
        'lda' or 'bertopic'.

    Returns
    -------
    np.ndarray
        Shape (n_docs, n_topics). Each row is a probability distribution.
    """
    if model_type == "lda":
        from src.topic_model import get_document_topics
        matrix = get_document_topics(model, corpus_or_texts)
        logger.info("LDA topic features — shape: %s", matrix.shape)
        return matrix

    elif model_type == "bertopic":
        topics, probs = model.transform(corpus_or_texts)
        probs_arr = np.asarray(probs) if probs is not None else None

        if probs_arr is not None and probs_arr.ndim == 2:
            # Full (n_docs, n_topics) distribution — use directly
            matrix = probs_arr.astype(np.float32)
        else:
            # probs is None or 1-D (per-doc assigned-topic probability only).
            # Fall back to one-hot encoding of the transform() topic assignments.
            topics = list(topics)
            valid = [t for t in topics if t >= 0]
            n_topics = (max(valid) + 1) if valid else 1
            matrix = np.zeros((len(corpus_or_texts), n_topics), dtype=np.float32)
            for i, t in enumerate(topics):
                if 0 <= t < n_topics:
                    matrix[i, t] = 1.0
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        logger.info("BERTopic topic features — shape: %s", matrix.shape)
        return matrix

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lda' or 'bertopic'.")


# ---------------------------------------------------------------------------
# 3. Structured features
# ---------------------------------------------------------------------------

def create_structured_features(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode demographic and clinical structured variables for prediction.

    Features extracted:
    - age (numeric, standardized)
    - gender (binary: M=1, F=0)
    - insurance (one-hot: Medicare, Medicaid, Other)
    - admission_type (one-hot: EMERGENCY, ELECTIVE, URGENT, EW EMER.)
    - los_days (log-transformed)
    - num_prior_admissions (count of earlier hadm_ids for same patient)
    - days_since_last_admission (days since previous discharge; -1 if first)
    - prior_expire_flag (hospital_expire_flag of most recent prior admission; 0 if first)

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset (output of data_loader.merge_dataset).

    Returns
    -------
    tuple
        (feature_matrix as np.ndarray, feature_names as list of str)
    """
    df = df.copy().reset_index(drop=True)

    features = pd.DataFrame(index=df.index)

    # --- Age ---
    age_col = "anchor_age" if "anchor_age" in df.columns else None
    if age_col:
        features["age"] = df[age_col].fillna(df[age_col].median()).astype(float)
    else:
        features["age"] = 65.0   # median fallback
        logger.warning("anchor_age not found — using placeholder 65.")

    # --- Gender ---
    features["gender_male"] = (df["gender"].str.upper() == "M").astype(int) \
        if "gender" in df.columns else 0

    # --- Insurance (one-hot) ---
    insurance_categories = ["Medicare", "Medicaid", "Other"]
    if "insurance" in df.columns:
        ins = df["insurance"].fillna("Other")
        for cat in insurance_categories:
            features[f"insurance_{cat.lower()}"] = (ins == cat).astype(int)
    else:
        for cat in insurance_categories:
            features[f"insurance_{cat.lower()}"] = 0

    # --- Admission type (one-hot) ---
    adm_types = ["EMERGENCY", "ELECTIVE", "URGENT"]
    if "admission_type" in df.columns:
        adm = df["admission_type"].fillna("EMERGENCY").str.upper()
        # Collapse EW EMER. into EMERGENCY
        adm = adm.str.replace("EW EMER.", "EMERGENCY", regex=False)
        for atype in adm_types:
            features[f"admtype_{atype.lower()}"] = (adm == atype).astype(int)
    else:
        for atype in adm_types:
            features[f"admtype_{atype.lower()}"] = 0

    # --- Length of stay (log-transformed) ---
    if "los_days" in df.columns:
        features["log_los"] = np.log1p(df["los_days"].fillna(0).clip(lower=0))
    else:
        features["log_los"] = 0.0

    # --- Prior admissions & days since last admission ---
    features["num_prior_admissions"] = 0
    features["days_since_last_admission"] = -1
    features["prior_expire_flag"] = 0

    if "subject_id" in df.columns and "admittime" in df.columns:
        df["admittime"] = pd.to_datetime(df["admittime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])

        for subj_id, grp in df.groupby("subject_id"):
            grp = grp.sort_values("admittime")
            idx_list = grp.index.tolist()
            for rank, idx in enumerate(idx_list):
                features.at[idx, "num_prior_admissions"] = rank
                if rank > 0:
                    prev_idx = idx_list[rank - 1]
                    gap = (grp.loc[idx, "admittime"] - grp.loc[prev_idx, "dischtime"]).days
                    features.at[idx, "days_since_last_admission"] = max(gap, 0)
                    prev_flag = df.loc[prev_idx, "hospital_expire_flag"] \
                        if "hospital_expire_flag" in df.columns else 0
                    features.at[idx, "prior_expire_flag"] = int(prev_flag)

    # --- Standardize numeric columns ---
    numeric_cols = ["age", "log_los", "num_prior_admissions", "days_since_last_admission"]
    for col in numeric_cols:
        col_data = features[col].astype(float)
        std = col_data.std()
        if std > 0:
            features[col] = (col_data - col_data.mean()) / std

    feature_names = list(features.columns)
    matrix = features.values.astype(np.float32)

    logger.info(
        "Structured features — shape: %s, features: %s", matrix.shape, feature_names
    )
    return matrix, feature_names


# ---------------------------------------------------------------------------
# 4. Text statistics features
# ---------------------------------------------------------------------------

def create_text_statistics_features(
    texts: List[str],
    original_texts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract surface-level text statistics from clinical notes.

    Features:
    - note_length_words    : word count of cleaned text
    - num_negations        : count of negation cues (no, not, denies, without, negative)
    - num_section_headers  : count of recognized section headers in the original note
    - avg_word_length      : average characters per word (proxy for medical terminology density)
    - type_token_ratio     : unique words / total words (lexical richness)

    Parameters
    ----------
    texts : list of str
        Cleaned (pre-processed) text strings.
    original_texts : list of str, optional
        Raw original note texts — used to count section headers.
        If None, section counts are skipped.

    Returns
    -------
    pd.DataFrame
        One row per document, one column per feature.
    """
    import re

    NEGATION_PATTERN = re.compile(
        r"\b(no|not|denies|without|negative|absent|none|never|neither|nor)\b",
        re.IGNORECASE,
    )

    SECTION_HEADERS = [
        "chief complaint", "history of present illness", "past medical history",
        "social history", "family history", "physical exam", "pertinent results",
        "brief hospital course", "medications on admission", "discharge medications",
        "discharge disposition", "discharge diagnosis", "discharge condition",
        "discharge instructions", "followup instructions",
    ]

    rows = []
    for i, text in enumerate(texts):
        words = str(text).split()
        n_words = len(words)

        negations = len(NEGATION_PATTERN.findall(text))

        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / n_words if n_words > 0 else 0.0

        avg_word_len = np.mean([len(w) for w in words]) if words else 0.0

        # Section header count from original text if available
        if original_texts is not None:
            orig = original_texts[i].lower()
            section_count = sum(1 for h in SECTION_HEADERS if h in orig)
        else:
            section_count = 0

        rows.append({
            "note_length_words": n_words,
            "num_negations": negations,
            "num_section_headers": section_count,
            "avg_word_length": round(avg_word_len, 3),
            "type_token_ratio": round(ttr, 4),
        })

    stats_df = pd.DataFrame(rows)

    # Standardize
    for col in stats_df.columns:
        std = stats_df[col].std()
        if std > 0:
            stats_df[col] = (stats_df[col] - stats_df[col].mean()) / std

    logger.info(
        "Text statistics features — shape: %s, features: %s",
        stats_df.shape, list(stats_df.columns),
    )
    return stats_df


# ---------------------------------------------------------------------------
# 5. Combine all feature sets
# ---------------------------------------------------------------------------

def combine_features(
    feature_sets: Dict[str, object],
) -> Tuple[object, List[str]]:
    """
    Horizontally stack any combination of sparse and dense feature matrices.

    Parameters
    ----------
    feature_sets : dict
        Mapping of feature_set_name -> matrix (scipy sparse or np.ndarray or pd.DataFrame).
        Example:
            {
                "tfidf": tfidf_matrix,           # sparse
                "topics": topic_matrix,           # dense ndarray
                "structured": structured_matrix,  # dense ndarray
                "text_stats": stats_df,           # DataFrame
            }

    Returns
    -------
    tuple
        (combined_matrix, feature_names_list)
        combined_matrix is scipy sparse if any input was sparse, else np.ndarray.
    """
    matrices = []
    all_feature_names = []

    for name, mat in feature_sets.items():
        if mat is None:
            logger.warning("Skipping feature set '%s' — None provided.", name)
            continue

        # Normalise type to scipy sparse or np.ndarray
        if isinstance(mat, pd.DataFrame):
            cols = [f"{name}__{c}" for c in mat.columns]
            mat = csr_matrix(mat.values.astype(np.float32))
            all_feature_names.extend(cols)
        elif isinstance(mat, np.ndarray):
            n_cols = mat.shape[1] if mat.ndim == 2 else 1
            all_feature_names.extend([f"{name}__{i}" for i in range(n_cols)])
            mat = csr_matrix(mat.astype(np.float32))
        elif issparse(mat):
            n_cols = mat.shape[1]
            all_feature_names.extend([f"{name}__{i}" for i in range(n_cols)])
        else:
            raise TypeError(f"Unsupported matrix type for '{name}': {type(mat)}")

        matrices.append(mat)
        logger.info("  Added '%s' — shape: %s", name, mat.shape)

    if not matrices:
        raise ValueError("No valid feature sets provided.")

    combined = hstack(matrices, format="csr")
    logger.info("Combined feature matrix — shape: %s", combined.shape)
    return combined, all_feature_names


# ---------------------------------------------------------------------------
# 6. Full feature engineering pipeline
# ---------------------------------------------------------------------------

def build_feature_sets(
    processed_df: pd.DataFrame,
    lda_results: Optional[dict] = None,
    bertopic_model=None,
    config: Optional[dict] = None,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """
    Build all feature sets from the processed DataFrame.

    Calls each feature-creation function and organises results into a dict
    that maps feature_type -> {'X': matrix, 'feature_names': [...], ...}

    Parameters
    ----------
    processed_df : pd.DataFrame
        Output of preprocess.build_preprocessing_pipeline() merged with
        data_loader.merge_dataset() data. Must have columns:
        'tokens', 'cleaned_text', 'text', and all structured columns.
    lda_results : dict, optional
        Output of topic_model.run_lda_pipeline(). If None, topic features
        are skipped.
    bertopic_model : BERTopic model, optional
        Fitted BERTopic. If None, BERTopic topic features are skipped.
    config : dict, optional
        Loaded config.yaml.
    embedding_matrix : np.ndarray, optional
        Pre-computed clinical embeddings of shape (n_docs, hidden_size).
        Generated by src.embeddings.embed_texts(). If None, embedding
        features are skipped.

    Returns
    -------
    dict
        {
          'tfidf':            {'X': sparse_matrix, 'vectorizer': ..., 'names': [...]},
          'topic_lda':        {'X': ndarray,        'names': [...]},
          'structured':       {'X': ndarray,        'names': [...]},
          'text_stats':       {'X': ndarray,        'names': [...]},
          'combined':         {'X': sparse_matrix,  'names': [...]},
          'vectorizer':       fitted TfidfVectorizer (for later transform),
          'label':            np.ndarray of readmission labels,
        }
    """
    cfg = (config or {})
    tfidf_cfg = cfg.get("prediction", {}).get("tfidf", {})

    result = {}

    # ---- Labels ----
    label_col = cfg.get("prediction", {}).get("target", "readmission_30day")
    if label_col in processed_df.columns:
        eligible_mask = processed_df[label_col] >= 0
        labels = processed_df.loc[eligible_mask, label_col].values.astype(int)
        df = processed_df[eligible_mask].reset_index(drop=True)
    else:
        logger.warning("Label column '%s' not found — using all rows.", label_col)
        df = processed_df.reset_index(drop=True)
        labels = None

    result["label"] = labels
    result["eligible_df"] = df

    # ---- TF-IDF ----
    logger.info("Building TF-IDF features ...")
    token_strings = df["tokens"].apply(
        lambda t: " ".join(t) if isinstance(t, list) else str(t)
    ).tolist()

    vectorizer, tfidf_matrix = create_tfidf_features(
        token_strings,
        max_features=tfidf_cfg.get("max_features", 5000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=tfidf_cfg.get("min_df", 5),
        max_df=tfidf_cfg.get("max_df", 0.95),
    )
    result["tfidf"] = {
        "X": tfidf_matrix,
        "vectorizer": vectorizer,
        "names": vectorizer.get_feature_names_out().tolist(),
    }
    result["vectorizer"] = vectorizer

    # ---- LDA topic features ----
    if lda_results is not None:
        logger.info("Building LDA topic features ...")
        lda_model = lda_results["best_model"]
        corpus = lda_results["corpus"]

        # Rebuild corpus for eligible subset if sizes differ
        if len(corpus) != len(df):
            from gensim.corpora import Dictionary
            dictionary = lda_results["dictionary"]
            tokens_list = df["tokens"].apply(
                lambda t: t if isinstance(t, list) else []
            ).tolist()
            corpus_eligible = [dictionary.doc2bow(t) for t in tokens_list]
        else:
            corpus_eligible = corpus

        topic_matrix = create_topic_features(lda_model, corpus_eligible, model_type="lda")
        n_topics = topic_matrix.shape[1]
        topic_labels = lda_results.get("topic_labels", {})
        names = [
            f"topic_lda_{i}_{topic_labels.get(i, '').replace(' ', '_').replace('/', '_')}"
            for i in range(n_topics)
        ]
        result["topic_lda"] = {"X": topic_matrix, "names": names}
    else:
        logger.info("Skipping LDA topic features (no lda_results provided).")
        result["topic_lda"] = None

    # ---- BERTopic features ----
    if bertopic_model is not None:
        logger.info("Building BERTopic features ...")
        bt_matrix = create_topic_features(
            bertopic_model, token_strings, model_type="bertopic"
        )
        result["topic_bertopic"] = {
            "X": bt_matrix,
            "names": [f"topic_bt_{i}" for i in range(bt_matrix.shape[1])],
        }
    else:
        result["topic_bertopic"] = None

    # ---- Clinical embeddings ----
    if embedding_matrix is not None:
        # Align to eligible subset if sizes differ
        if embedding_matrix.shape[0] != len(df):
            logger.warning(
                "Embedding matrix rows (%d) != eligible rows (%d) — "
                "attempting to slice with eligible mask.",
                embedding_matrix.shape[0], len(df),
            )
            if embedding_matrix.shape[0] == len(processed_df):
                embedding_matrix = embedding_matrix[eligible_mask.values] \
                    if hasattr(eligible_mask, 'values') else embedding_matrix[eligible_mask]
            else:
                logger.error("Cannot align embedding matrix — skipping.")
                embedding_matrix = None

    if embedding_matrix is not None:
        emb_cfg = cfg.get("embeddings", {})
        use_reduction = emb_cfg.get("reduce_dims", False)

        if use_reduction:
            from src.embeddings import reduce_embeddings
            n_components = emb_cfg.get("n_components", 50)
            reduction_method = emb_cfg.get("reduction_method", "pca")
            reduced, _ = reduce_embeddings(
                embedding_matrix, n_components=n_components, method=reduction_method,
            )
            emb_dim = reduced.shape[1]
            result["embeddings"] = {
                "X": reduced,
                "names": [f"emb_{i}" for i in range(emb_dim)],
            }
            logger.info("Clinical embeddings (reduced) — shape: %s", reduced.shape)
        else:
            emb_dim = embedding_matrix.shape[1]
            result["embeddings"] = {
                "X": embedding_matrix,
                "names": [f"emb_{i}" for i in range(emb_dim)],
            }
            logger.info("Clinical embeddings — shape: %s", embedding_matrix.shape)
    else:
        logger.info("Skipping clinical embeddings (no embedding_matrix provided).")
        result["embeddings"] = None

    # ---- Structured features ----
    logger.info("Building structured features ...")
    struct_matrix, struct_names = create_structured_features(df)
    result["structured"] = {"X": struct_matrix, "names": struct_names}

    # ---- Text statistics ----
    logger.info("Building text statistics features ...")
    original_texts = df["text"].tolist() if "text" in df.columns else None
    stats_df = create_text_statistics_features(token_strings, original_texts=original_texts)
    result["text_stats"] = {
        "X": stats_df.values.astype(np.float32),
        "names": [f"textstats__{c}" for c in stats_df.columns],
    }

    # ---- Combined ----
    logger.info("Building combined feature set ...")
    to_combine = {"tfidf": tfidf_matrix}
    if result["topic_lda"] is not None:
        to_combine["topics"] = result["topic_lda"]["X"]
    if result["embeddings"] is not None:
        to_combine["embeddings"] = result["embeddings"]["X"]
    to_combine["structured"] = struct_matrix
    to_combine["text_stats"] = stats_df

    combined_matrix, combined_names = combine_features(to_combine)
    result["combined"] = {"X": combined_matrix, "names": combined_names}

    # Summary
    logger.info("Feature sets ready:")
    for key in ["tfidf", "topic_lda", "embeddings", "structured", "text_stats", "combined"]:
        entry = result.get(key)
        if entry:
            shape = entry["X"].shape if hasattr(entry["X"], "shape") else "N/A"
            logger.info("  %-15s shape: %s", key, shape)

    return result
