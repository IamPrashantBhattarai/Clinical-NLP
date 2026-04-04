"""
topic_model.py
Purpose: Implement and evaluate LDA and BERTopic topic models on clinical notes.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clinical keyword vocabulary for topic labeling
# ---------------------------------------------------------------------------

CLINICAL_LABEL_KEYWORDS = {
    "Cardiovascular": {
        "cardiac", "heart", "coronary", "artery", "ejection", "fraction",
        "atrial", "fibrillation", "myocardial", "infarction", "bnp",
        "troponin", "echocardiogram", "diuresis", "furosemide", "metoprolol",
        "pacemaker", "valve", "aortic", "mitral", "ventricular", "systolic",
        "diastolic", "cardiomyopathy", "stent", "catheterization", "heparin",
    },
    "Respiratory / Pulmonary": {
        "respiratory", "pulmonary", "oxygen", "breath", "copd", "inhaler",
        "albuterol", "nebulization", "intubation", "ventilator", "bronchial",
        "pneumonia", "infiltrate", "effusion", "pleural", "hypoxia", "saturation",
        "dyspnea", "wheeze", "tiotropium", "steroid", "solumedrol", "prednisone",
        "bronchoscopy", "thoracentesis", "lung",
    },
    "Infection / Sepsis": {
        "infection", "sepsis", "septic", "antibiotic", "bacteremia", "bacteria",
        "culture", "blood_culture", "vancomycin", "piperacillin", "cefepime",
        "levofloxacin", "fever", "leukocytosis", "wbc", "wound", "abscess",
        "cellulitis", "pneumonia", "uti", "urinary", "mrsa", "pseudomonas",
        "streptococcus", "klebsiella", "ceftriaxone",
    },
    "Renal / Kidney": {
        "renal", "kidney", "creatinine", "nephrology", "dialysis", "hdialysis",
        "electrolyte", "potassium", "sodium", "fluid", "hydration", "urine",
        "oliguria", "foley", "glomerular", "proteinuria", "nephropathy",
        "prerenal", "azotemia", "bicarbonate", "acidosis", "alkalosis",
        "magnesium", "phosphorus",
    },
    "Neurological": {
        "neuro", "stroke", "cerebral", "infarct", "hemorrhage", "cranial",
        "seizure", "encephalopathy", "delirium", "confusion", "altered",
        "mental", "status", "aphasia", "hemiplegia", "weakness", "tpa",
        "mri", "ct_head", "neurology", "antiplatelet", "clopidogrel",
        "warfarin", "anticoagulation",
    },
    "Gastrointestinal": {
        "gastrointestinal", "abdominal", "bowel", "liver", "hepatic", "cirrhosis",
        "ascites", "varices", "bleed", "egd", "colonoscopy", "endoscopy",
        "pancreatitis", "lipase", "amylase", "nausea", "vomiting", "diarrhea",
        "constipation", "obstruction", "hernia", "cholecystitis", "appendicitis",
        "ppi", "omeprazole", "gastric", "colon", "rectal", "hemorrhoid",
    },
    "Endocrine / Metabolic": {
        "diabetes", "insulin", "glucose", "ketoacidosis", "dka", "hyperglycemia",
        "hypoglycemia", "thyroid", "hypothyroidism", "hyperthyroidism",
        "endocrinology", "metformin", "a1c", "hemoglobin", "cortisol",
        "adrenal", "electrolyte", "sodium", "potassium", "calcium",
    },
    "Surgical / Postoperative": {
        "surgery", "surgical", "operative", "postoperative", "incision",
        "wound", "drain", "anastomosis", "laparotomy", "laparoscopic",
        "orthopedic", "fracture", "fixation", "arthroplasty", "hip",
        "knee", "amputation", "vascular", "bypass", "graft",
    },
    "Psychiatric / Behavioral": {
        "psychiatric", "depression", "anxiety", "psychosis", "schizophrenia",
        "bipolar", "suicidal", "overdose", "substance", "alcohol", "withdrawal",
        "benzodiazepine", "antipsychotic", "haloperidol", "quetiapine",
        "ssri", "serotonin", "behavioral", "dementia",
    },
}


# ---------------------------------------------------------------------------
# 1. Train LDA
# ---------------------------------------------------------------------------

def train_lda(
    corpus: list,
    dictionary,
    num_topics: int,
    passes: int = 15,
    iterations: int = 100,
    chunksize: int = 2000,
    alpha: str = "auto",
    eta: str = "auto",
    random_seed: int = 42,
):
    """
    Train a gensim LdaMulticore model.

    Parameters
    ----------
    corpus : list
        Gensim BoW corpus (list of (id, count) lists).
    dictionary : gensim.corpora.Dictionary
        Gensim Dictionary mapping token -> id.
    num_topics : int
        Number of topics to discover.
    passes : int
        Number of full passes over the corpus.
    iterations : int
        Max number of EM iterations per pass.
    chunksize : int
        Number of documents per training chunk.
    alpha : str or float
        Document-topic prior. 'auto' = learned from data.
    eta : str or float
        Topic-word prior. 'auto' = learned from data.
    random_seed : int
        For reproducibility.

    Returns
    -------
    gensim.models.LdaMulticore
        Trained LDA model.
    """
    # LdaMulticore does not support alpha/eta='auto' — use LdaModel in that case.
    auto_tune = (alpha == "auto" or eta == "auto")

    if auto_tune:
        from gensim.models import LdaModel
        logger.info(
            "Training LdaModel (alpha/eta=auto): num_topics=%d, passes=%d ...",
            num_topics, passes,
        )
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            chunksize=chunksize,
            alpha=alpha,
            eta=eta,
            random_state=random_seed,
        )
    else:
        from gensim.models import LdaMulticore
        logger.info(
            "Training LdaMulticore: num_topics=%d, passes=%d ...",
            num_topics, passes,
        )
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            chunksize=chunksize,
            alpha=alpha,
            eta=eta,
            random_state=random_seed,
            per_word_topics=False,
        )

    logger.info("LDA training complete.")
    return model


# ---------------------------------------------------------------------------
# 2. Evaluate coherence
# ---------------------------------------------------------------------------

def evaluate_coherence(
    models_dict: Dict[int, object],
    corpus: list,
    dictionary,
    texts: List[List[str]],
    coherence: str = "c_v",
) -> Dict[int, float]:
    """
    Calculate coherence score for each trained LDA model.

    Parameters
    ----------
    models_dict : dict
        {num_topics: trained_lda_model}
    corpus : list
        Gensim BoW corpus.
    dictionary : gensim.corpora.Dictionary
    texts : list of list of str
        Tokenized documents (needed for c_v coherence).
    coherence : str
        Coherence measure: 'c_v', 'u_mass', 'c_uci', 'c_npmi'.

    Returns
    -------
    dict
        {num_topics: coherence_score}
    """
    from gensim.models.coherencemodel import CoherenceModel

    scores = {}
    for n_topics, model in models_dict.items():
        logger.info("Evaluating coherence (%s) for num_topics=%d ...", coherence, n_topics)
        cm = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence=coherence,
        )
        score = cm.get_coherence()
        scores[n_topics] = round(score, 4)
        logger.info("  num_topics=%d -> coherence=%.4f", n_topics, score)
    return scores


# ---------------------------------------------------------------------------
# 3. Find optimal number of topics
# ---------------------------------------------------------------------------

def find_optimal_topics(
    corpus: list,
    dictionary,
    texts: List[List[str]],
    topic_range: List[int] = None,
    passes: int = 15,
    iterations: int = 100,
    random_seed: int = 42,
) -> Tuple[int, object, Dict[int, float]]:
    """
    Train LDA for each value in topic_range and select the best by coherence.

    Parameters
    ----------
    corpus : list
        Gensim BoW corpus.
    dictionary : gensim.corpora.Dictionary
    texts : list of list of str
        Tokenized documents.
    topic_range : list of int
        Topic counts to try. Defaults to [5, 10, 15, 20].
    passes : int
    iterations : int
    random_seed : int

    Returns
    -------
    tuple
        (best_num_topics, best_model, coherence_scores_dict)
    """
    if topic_range is None:
        topic_range = [5, 10, 15, 20]

    models = {}
    for n in topic_range:
        models[n] = train_lda(
            corpus, dictionary, num_topics=n,
            passes=passes, iterations=iterations,
            random_seed=random_seed,
        )

    coherence_scores = evaluate_coherence(models, corpus, dictionary, texts)

    best_n = max(coherence_scores, key=coherence_scores.get)
    logger.info(
        "Optimal num_topics=%d (coherence=%.4f)", best_n, coherence_scores[best_n]
    )
    return best_n, models[best_n], coherence_scores


# ---------------------------------------------------------------------------
# 4. Get topic words
# ---------------------------------------------------------------------------

def get_topic_words(
    model,
    num_words: int = 15,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Extract top words and their weights for each topic.

    Parameters
    ----------
    model : trained LDA model
    num_words : int
        Number of top words per topic.

    Returns
    -------
    dict
        {topic_id: [(word, weight), ...]}
    """
    topic_words = {}
    for topic_id in range(model.num_topics):
        top = model.show_topic(topic_id, topn=num_words)
        topic_words[topic_id] = [(word, round(weight, 4)) for word, weight in top]
    return topic_words


# ---------------------------------------------------------------------------
# 5. Label topics
# ---------------------------------------------------------------------------

def label_topics(
    topic_words: Dict[int, List[Tuple[str, float]]],
) -> Dict[int, str]:
    """
    Suggest human-readable clinical labels for each topic based on top words.

    Scoring: count how many top words match each clinical category's keyword set.
    The category with the highest match count wins.
    Falls back to 'Topic N' if no keywords match.

    Parameters
    ----------
    topic_words : dict
        {topic_id: [(word, weight), ...]}  — output of get_topic_words().

    Returns
    -------
    dict
        {topic_id: suggested_label}
    """
    labels = {}
    for topic_id, word_weight_list in topic_words.items():
        words = {w.lower() for w, _ in word_weight_list}
        scores = {}
        for category, keywords in CLINICAL_LABEL_KEYWORDS.items():
            scores[category] = len(words & keywords)

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score == 0:
            labels[topic_id] = f"Topic {topic_id}"
        else:
            labels[topic_id] = best_category

    # Deduplicate: if multiple topics get the same label, suffix them
    seen: Dict[str, int] = {}
    deduped = {}
    for tid, label in labels.items():
        if label == f"Topic {tid}":
            deduped[tid] = label
            continue
        count = seen.get(label, 0)
        seen[label] = count + 1
        deduped[tid] = label if count == 0 else f"{label} ({count + 1})"
    return deduped


# ---------------------------------------------------------------------------
# 6. Get document-topic distributions
# ---------------------------------------------------------------------------

def get_document_topics(
    model,
    corpus: list,
) -> np.ndarray:
    """
    Compute topic probability distribution for every document.

    Parameters
    ----------
    model : trained LDA model
    corpus : list
        Gensim BoW corpus.

    Returns
    -------
    np.ndarray
        Shape (n_docs, n_topics). Each row sums to ~1.
    """
    n_docs = len(corpus)
    n_topics = model.num_topics
    doc_topic_matrix = np.zeros((n_docs, n_topics), dtype=np.float32)

    for i, bow in enumerate(corpus):
        topic_dist = model.get_document_topics(bow, minimum_probability=0.0)
        for topic_id, prob in topic_dist:
            doc_topic_matrix[i, topic_id] = prob

    return doc_topic_matrix


# ---------------------------------------------------------------------------
# 7. Topic summary DataFrame
# ---------------------------------------------------------------------------

def topics_to_dataframe(
    topic_words: Dict[int, List[Tuple[str, float]]],
    topic_labels: Dict[int, str],
    num_words: int = 10,
) -> pd.DataFrame:
    """
    Create a tidy DataFrame summarising topics — useful for display in notebooks.

    Parameters
    ----------
    topic_words : dict
        {topic_id: [(word, weight), ...]}
    topic_labels : dict
        {topic_id: label}
    num_words : int
        How many top words to include per row.

    Returns
    -------
    pd.DataFrame
        Columns: topic_id, label, top_words, top_word_weights
    """
    rows = []
    for tid, words in topic_words.items():
        top = words[:num_words]
        rows.append({
            "topic_id": tid,
            "label": topic_labels.get(tid, f"Topic {tid}"),
            "top_words": ", ".join(w for w, _ in top),
            "top_word_weights": [round(wt, 4) for _, wt in top],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. BERTopic (transformer-based)
# ---------------------------------------------------------------------------

def train_bertopic(
    texts: List[str],
    config: Optional[dict] = None,
):
    """
    Fit a BERTopic model with BioBERT embeddings, UMAP, and HDBSCAN.

    Falls back gracefully to sentence-transformers 'all-MiniLM-L6-v2' if
    BioBERT cannot be loaded (e.g. limited connectivity or compute).

    Parameters
    ----------
    texts : list of str
        Cleaned (not tokenized) document strings.
    config : dict, optional
        Loaded config.yaml dict for BERTopic hyperparameters.

    Returns
    -------
    tuple
        (bertopic_model, topics_list, probabilities_array)
        Returns (None, None, None) if BERTopic cannot be imported.
    """
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        logger.warning("BERTopic dependencies not available: %s. Skipping.", e)
        return None, None, None

    bt_cfg = (config or {}).get("topic_modeling", {}).get("bertopic", {})
    embedding_model_name = bt_cfg.get("embedding_model", "dmis-lab/biobert-base-cased-v1.2")
    min_topic_size = bt_cfg.get("min_topic_size", 50)
    nr_topics = bt_cfg.get("nr_topics", "auto")
    top_n_words = bt_cfg.get("top_n_words", 15)
    umap_n_neighbors = bt_cfg.get("umap_n_neighbors", 15)
    umap_n_components = bt_cfg.get("umap_n_components", 5)
    umap_min_dist = bt_cfg.get("umap_min_dist", 0.0)
    hdbscan_min_cluster = bt_cfg.get("hdbscan_min_cluster_size", 50)

    logger.info("Loading embedding model: %s ...", embedding_model_name)
    try:
        embedding_model = SentenceTransformer(embedding_model_name)
    except Exception:
        fallback = "all-MiniLM-L6-v2"
        logger.warning(
            "Could not load %s — falling back to %s", embedding_model_name, fallback
        )
        embedding_model = SentenceTransformer(fallback)

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        top_n_words=top_n_words,
        verbose=True,
    )

    logger.info("Fitting BERTopic on %d documents ...", len(texts))
    topics, probs = topic_model.fit_transform(texts)
    n_topics_found = len(set(topics)) - (1 if -1 in topics else 0)
    logger.info("BERTopic found %d topics.", n_topics_found)
    return topic_model, topics, probs


# ---------------------------------------------------------------------------
# 9. Compare LDA vs BERTopic
# ---------------------------------------------------------------------------

def compare_models(
    lda_topic_words: Dict[int, List[Tuple[str, float]]],
    bertopic_model=None,
    texts: Optional[List[List[str]]] = None,
    dictionary=None,
    corpus: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compare LDA and BERTopic on coherence and topic diversity.

    Topic diversity = % of unique words across all top-10 topic words.
    High diversity means topics are more distinct from each other.

    Parameters
    ----------
    lda_topic_words : dict
        {topic_id: [(word, weight), ...]} from get_topic_words().
    bertopic_model : BERTopic model or None
    texts : list of list of str
        Tokenized texts (for LDA coherence).
    dictionary : gensim Dictionary (for LDA coherence).
    corpus : list (for LDA coherence).

    Returns
    -------
    pd.DataFrame
        Comparison table with rows for LDA (and BERTopic if available).
    """
    rows = []

    # --- LDA stats ---
    lda_all_words = [w for words in lda_topic_words.values() for w, _ in words[:10]]
    lda_diversity = round(len(set(lda_all_words)) / len(lda_all_words), 4) if lda_all_words else 0.0
    n_lda_topics = len(lda_topic_words)

    lda_coherence = None
    if texts and dictionary and corpus:
        try:
            from gensim.models.coherencemodel import CoherenceModel
            # Reconstruct topic word lists for CoherenceModel
            topics_as_lists = [[w for w, _ in wl[:10]] for wl in lda_topic_words.values()]
            cm = CoherenceModel(
                topics=topics_as_lists,
                texts=texts,
                dictionary=dictionary,
                coherence="c_v",
            )
            lda_coherence = round(cm.get_coherence(), 4)
        except Exception as e:
            logger.warning("Could not compute LDA coherence: %s", e)

    rows.append({
        "model": "LDA",
        "num_topics": n_lda_topics,
        "coherence_c_v": lda_coherence,
        "topic_diversity": lda_diversity,
        "notes": "gensim LdaMulticore",
    })

    # --- BERTopic stats ---
    if bertopic_model is not None:
        try:
            bt_topic_info = bertopic_model.get_topic_info()
            n_bt_topics = len(bt_topic_info[bt_topic_info["Topic"] != -1])
            bt_all_words = []
            for tid in bt_topic_info[bt_topic_info["Topic"] != -1]["Topic"]:
                words = bertopic_model.get_topic(tid)
                bt_all_words.extend([w for w, _ in words[:10]])
            bt_diversity = round(len(set(bt_all_words)) / len(bt_all_words), 4) if bt_all_words else 0.0

            rows.append({
                "model": "BERTopic",
                "num_topics": n_bt_topics,
                "coherence_c_v": None,   # BERTopic doesn't use gensim coherence natively
                "topic_diversity": bt_diversity,
                "notes": "BioBERT + UMAP + HDBSCAN",
            })
        except Exception as e:
            logger.warning("Could not extract BERTopic stats: %s", e)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. Topic-readmission association test
# ---------------------------------------------------------------------------

def test_topic_readmission_association(
    doc_topic_matrix: np.ndarray,
    readmission_labels: np.ndarray,
    topic_labels: Dict[int, str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test whether each topic's prevalence differs between readmitted vs
    non-readmitted patients using a Mann-Whitney U test.

    Parameters
    ----------
    doc_topic_matrix : np.ndarray
        Shape (n_docs, n_topics) — output of get_document_topics().
    readmission_labels : np.ndarray
        Binary labels (0/1). Must have same length as doc_topic_matrix.
    topic_labels : dict
        {topic_id: label}
    alpha : float
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        Columns: topic_id, label, mean_readmitted, mean_not_readmitted,
                 u_statistic, p_value, significant
        Sorted by p_value ascending.
    """
    from scipy.stats import mannwhitneyu

    labels = np.array(readmission_labels)
    readmitted_idx = labels == 1
    not_readmitted_idx = labels == 0

    rows = []
    for tid in range(doc_topic_matrix.shape[1]):
        topic_scores = doc_topic_matrix[:, tid]
        group_pos = topic_scores[readmitted_idx]
        group_neg = topic_scores[not_readmitted_idx]

        if len(group_pos) == 0 or len(group_neg) == 0:
            continue

        u_stat, p_val = mannwhitneyu(group_pos, group_neg, alternative="two-sided")
        rows.append({
            "topic_id": tid,
            "label": topic_labels.get(tid, f"Topic {tid}"),
            "mean_readmitted": round(float(group_pos.mean()), 4),
            "mean_not_readmitted": round(float(group_neg.mean()), 4),
            "u_statistic": round(u_stat, 2),
            "p_value": round(p_val, 4),
            "significant": p_val < alpha,
        })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 11. Convenience: full LDA pipeline in one call
# ---------------------------------------------------------------------------

def run_lda_pipeline(
    tokenized_docs: List[List[str]],
    config: Optional[dict] = None,
) -> dict:
    """
    Run the full LDA pipeline: build corpus, grid-search topics, label, get
    document distributions.

    Parameters
    ----------
    tokenized_docs : list of list of str
        Preprocessed, tokenized documents.
    config : dict, optional
        Loaded config.yaml dict.

    Returns
    -------
    dict with keys:
        dictionary, corpus, best_model, best_num_topics,
        coherence_scores, topic_words, topic_labels,
        doc_topic_matrix, topics_df
    """
    from gensim.corpora import Dictionary

    lda_cfg = (config or {}).get("topic_modeling", {}).get("lda", {})
    topic_range = lda_cfg.get("num_topics", [5, 10, 15, 20])
    passes = lda_cfg.get("passes", 15)
    iterations = lda_cfg.get("iterations", 100)
    random_seed = (config or {}).get("data", {}).get("random_seed", 42)

    # Build dictionary and corpus
    logger.info("Building dictionary and corpus ...")
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.95)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    logger.info("Dictionary: %d tokens | Corpus: %d docs", len(dictionary), len(corpus))

    # Grid search
    best_n, best_model, coherence_scores = find_optimal_topics(
        corpus=corpus,
        dictionary=dictionary,
        texts=tokenized_docs,
        topic_range=topic_range,
        passes=passes,
        iterations=iterations,
        random_seed=random_seed,
    )

    # Extract topic words and labels
    topic_words = get_topic_words(best_model, num_words=15)
    topic_labels = label_topics(topic_words)
    doc_topic_matrix = get_document_topics(best_model, corpus)
    topics_df = topics_to_dataframe(topic_words, topic_labels)

    logger.info("LDA pipeline complete. Best: %d topics.", best_n)
    for tid, label in topic_labels.items():
        words_str = ", ".join(w for w, _ in topic_words[tid][:6])
        logger.info("  Topic %2d [%s]: %s ...", tid, label, words_str)

    return {
        "dictionary": dictionary,
        "corpus": corpus,
        "best_model": best_model,
        "best_num_topics": best_n,
        "coherence_scores": coherence_scores,
        "topic_words": topic_words,
        "topic_labels": topic_labels,
        "doc_topic_matrix": doc_topic_matrix,
        "topics_df": topics_df,
    }
