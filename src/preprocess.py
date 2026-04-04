"""
preprocess.py
Purpose: Clean and preprocess clinical text for NLP analysis.
"""

import logging
import re
import string
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load heavy NLP models so import stays fast
# ---------------------------------------------------------------------------
_spacy_model = None
_nltk_ready = False


def _get_spacy(use_scispacy: bool = False):
    global _spacy_model
    if _spacy_model is None:
        try:
            if use_scispacy:
                import spacy
                _spacy_model = spacy.load("en_core_sci_sm")
                logger.info("Loaded scispacy model: en_core_sci_sm")
            else:
                import spacy
                _spacy_model = spacy.load("en_core_web_sm")
                logger.info("Loaded spacy model: en_core_web_sm")
        except OSError as e:
            raise OSError(
                f"spaCy model not found. Run: python -m spacy download en_core_web_sm\n{e}"
            )
    return _spacy_model


def _ensure_nltk():
    global _nltk_ready
    if not _nltk_ready:
        import nltk
        for resource in ["stopwords", "wordnet", "punkt", "punkt_tab", "averaged_perceptron_tagger"]:
            try:
                category = "tokenizers" if resource in ("punkt", "punkt_tab") else "corpora"
                nltk.data.find(f"{category}/{resource}")
            except LookupError:
                logger.info("Downloading NLTK resource: %s", resource)
                nltk.download(resource, quiet=True)
        _nltk_ready = True


# ---------------------------------------------------------------------------
# Custom medical stopwords (augment NLTK defaults)
# ---------------------------------------------------------------------------

DEFAULT_MEDICAL_STOPWORDS = {
    "patient", "pt", "history", "mg", "ml", "dr", "hospital",
    "admission", "discharge", "date", "year", "old", "day",
    "days", "time", "week", "month", "per", "given", "noted",
    "follow", "also", "including", "significant", "without",
    "well", "due", "placed", "started", "continued",
}

# Common section headers in MIMIC discharge summaries
SECTION_HEADERS = [
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Social History",
    "Family History",
    "Physical Exam",
    "Pertinent Results",
    "Brief Hospital Course",
    "Medications on Admission",
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
    "Allergies",
    "Attending",
]

# Compiled regex patterns (compiled once for performance)
_RE_DEID = re.compile(r"_{2,}")                                      # de-identification blanks
_RE_CHECKBOX = re.compile(r"\[\s*[xX]?\s*\]")                        # [ ] or [x]
_RE_DATE_MDY = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")           # MM/DD/YYYY
_RE_DATE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")                  # YYYY-MM-DD
_RE_PHONE = re.compile(r"\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}")         # phone / fax
_RE_LAB_VALUE = re.compile(r"\b\d+\.?\d*\s*(?:mg|ml|mcg|mmol|meq|mm|cm|kg|lb|units?)\b", re.IGNORECASE)
_RE_WHITESPACE = re.compile(r"[ \t]+")
_RE_NEWLINES = re.compile(r"\n{3,}")
_RE_PURE_NUMBER = re.compile(r"^\d+\.?\d*$")


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------

def clean_clinical_text(text: str) -> str:
    """
    Normalize a raw clinical note.

    Steps:
    - Remove de-identification markers (___).
    - Remove checkbox patterns ([ ] / [x]).
    - Remove dates (MM/DD/YYYY, YYYY-MM-DD).
    - Remove phone / fax numbers.
    - Remove lab value strings with units.
    - Lowercase.
    - Normalize whitespace.

    Parameters
    ----------
    text : str
        Raw clinical note text.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _RE_DEID.sub(" ", text)
    text = _RE_CHECKBOX.sub(" ", text)
    text = _RE_DATE_MDY.sub(" ", text)
    text = _RE_DATE_ISO.sub(" ", text)
    text = _RE_PHONE.sub(" ", text)
    text = _RE_LAB_VALUE.sub(" ", text)
    text = text.lower()
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_NEWLINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Section extraction
# ---------------------------------------------------------------------------

def extract_sections(text: str) -> Dict[str, str]:
    """
    Parse a discharge summary into named sections.

    Parameters
    ----------
    text : str
        Raw or lightly cleaned note text.

    Returns
    -------
    dict
        Mapping of section_name -> section_text.
        Unmatched leading text is stored under key 'preamble'.
    """
    sections: Dict[str, str] = {}
    # Build pattern: header followed by optional colon/newline
    header_pattern = re.compile(
        r"(?m)^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?\s*$",
        re.IGNORECASE,
    )

    parts = header_pattern.split(text)
    # parts layout: [pre_text, header1, body1, header2, body2, ...]
    if parts:
        preamble = parts[0].strip()
        if preamble:
            sections["preamble"] = preamble

    it = iter(parts[1:])
    for header, body in zip(it, it):
        sections[header.strip()] = body.strip()

    return sections


def remove_sections(text: str, sections_to_remove: List[str]) -> str:
    """
    Remove specified sections from a discharge summary.

    Parameters
    ----------
    text : str
        Full note text.
    sections_to_remove : list of str
        Section names to strip (e.g. 'Discharge Medications').

    Returns
    -------
    str
        Note text with those sections removed.
    """
    for section in sections_to_remove:
        pattern = re.compile(
            rf"(?im)^{re.escape(section)}\s*:?\s*\n(.*?)(?=\n[A-Z][A-Za-z ]+\s*:?\s*\n|\Z)",
            re.DOTALL,
        )
        text = pattern.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Tokenization
# ---------------------------------------------------------------------------

def _spacy_available() -> bool:
    """Check if spaCy and its models can actually be imported without error."""
    try:
        import spacy  # noqa: F401
        return True
    except Exception:
        return False


def _tokenize_with_nltk(
    text: str,
    stop_words: set,
) -> List[str]:
    """NLTK-based fallback tokenizer with WordNetLemmatizer."""
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    raw_tokens = word_tokenize(text)
    tokens = []
    for tok in raw_tokens:
        tok = tok.lower().strip()
        if (
            len(tok) <= 1
            or tok in stop_words
            or _RE_PURE_NUMBER.match(tok)
            or tok in string.punctuation
            or not tok.isalpha()
        ):
            continue
        tokens.append(lemmatizer.lemmatize(tok))
    return tokens


def tokenize_clinical(
    text: str,
    use_scispacy: bool = False,
    extra_stopwords: Optional[set] = None,
) -> List[str]:
    """
    Tokenize and lemmatize clinical text.

    Prefers spaCy (en_core_web_sm) or scispaCy (en_core_sci_sm) when available.
    Falls back to NLTK WordNetLemmatizer if spaCy cannot be loaded.

    Parameters
    ----------
    text : str
        Cleaned clinical text (lower-cased).
    use_scispacy : bool
        If True, attempt to use en_core_sci_sm (biomedical tokenizer).
    extra_stopwords : set, optional
        Additional domain-specific stopwords to remove.

    Returns
    -------
    list of str
        Filtered, lemmatized tokens.
    """
    _ensure_nltk()
    from nltk.corpus import stopwords as nltk_stopwords

    stop_words = set(nltk_stopwords.words("english"))
    stop_words |= DEFAULT_MEDICAL_STOPWORDS
    if extra_stopwords:
        stop_words |= set(extra_stopwords)

    if _spacy_available():
        nlp = _get_spacy(use_scispacy=use_scispacy)
        doc = nlp(text[:1_000_000])
        tokens = []
        for tok in doc:
            lemma = tok.lemma_.lower().strip()
            if (
                tok.is_stop
                or tok.is_punct
                or tok.is_space
                or len(lemma) <= 1
                or lemma in stop_words
                or _RE_PURE_NUMBER.match(lemma)
                or lemma in string.punctuation
            ):
                continue
            tokens.append(lemma)
        return tokens
    else:
        return _tokenize_with_nltk(text, stop_words)


# ---------------------------------------------------------------------------
# 4. Bigrams / Trigrams
# ---------------------------------------------------------------------------

def create_bigrams_trigrams(
    tokenized_docs: List[List[str]],
    min_count: int = 10,
    threshold: float = 50.0,
    scoring: str = "default",
) -> List[List[str]]:
    """
    Detect common bigrams and trigrams using gensim Phrases.

    Common clinical bigrams like 'heart_failure', 'blood_pressure' are joined
    with an underscore so they are treated as single tokens by topic models.

    Parameters
    ----------
    tokenized_docs : list of list of str
        Tokenized documents.
    min_count : int
        Minimum co-occurrence count to form a phrase.
    threshold : float
        Scoring threshold for phrase acceptance.

    Returns
    -------
    list of list of str
        Documents with bigrams/trigrams joined by '_'.
    """
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

    logger.info("Building bigram model ...")
    bigram_model = Phrases(
        tokenized_docs,
        min_count=min_count,
        threshold=threshold,
        scoring=scoring,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    bigram_docs = [bigram_model[doc] for doc in tokenized_docs]

    logger.info("Building trigram model ...")
    trigram_model = Phrases(
        bigram_docs,
        min_count=min_count,
        threshold=threshold,
        scoring=scoring,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    trigram_docs = [trigram_model[doc] for doc in bigram_docs]
    return trigram_docs


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    config: Optional[dict] = None,
    use_scispacy: bool = False,
    use_phrases: bool = True,
) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a DataFrame column.

    Steps:
    1. Remove unwanted sections (from config).
    2. Clean raw text.
    3. Tokenize + lemmatize.
    4. Detect bigrams / trigrams (optional).
    5. Add derived columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a text column.
    text_col : str
        Name of the column containing raw note text.
    config : dict, optional
        Loaded config.yaml dict; used for stopwords and section removal.
    use_scispacy : bool
        Use biomedical tokenizer.
    use_phrases : bool
        Detect bigrams/trigrams.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns:
        - 'cleaned_text'  : cleaned, lowercased string
        - 'tokens'        : list of lemmatized tokens
        - 'num_tokens'    : token count per document
    """
    pp_cfg = (config or {}).get("preprocessing", {})
    sections_to_remove = pp_cfg.get("remove_sections", [])
    extra_stopwords = set(pp_cfg.get("custom_stopwords", []))

    logger.info("Step 1/4 — removing unwanted sections from %d notes ...", len(df))
    if sections_to_remove:
        df = df.copy()
        df[text_col] = df[text_col].apply(
            lambda t: remove_sections(str(t), sections_to_remove)
        )

    logger.info("Step 2/4 — cleaning text ...")
    df["cleaned_text"] = df[text_col].apply(clean_clinical_text)

    # Filter out notes that are too short or too long
    min_len = pp_cfg.get("min_note_length", 100)
    max_len = pp_cfg.get("max_note_length", 50000)
    before = len(df)
    df = df[
        df["cleaned_text"].str.len().between(min_len, max_len)
    ].reset_index(drop=True)
    logger.info("Filtered %d notes outside length bounds — %d remaining", before - len(df), len(df))

    logger.info("Step 3/4 — tokenizing %d notes (use_scispacy=%s) ...", len(df), use_scispacy)
    df["tokens"] = df["cleaned_text"].apply(
        lambda t: tokenize_clinical(t, use_scispacy=use_scispacy, extra_stopwords=extra_stopwords)
    )

    if use_phrases:
        logger.info("Step 4/4 — detecting bigrams/trigrams ...")
        df["tokens"] = create_bigrams_trigrams(df["tokens"].tolist())
    else:
        logger.info("Step 4/4 — skipping phrase detection.")

    df["num_tokens"] = df["tokens"].apply(len)

    vocab = set(tok for toks in df["tokens"] for tok in toks)
    avg_tokens = df["num_tokens"].mean()
    logger.info(
        "Preprocessing complete — vocab size: %d, avg tokens/doc: %.1f",
        len(vocab), avg_tokens,
    )
    return df


# ---------------------------------------------------------------------------
# 6. Document-term matrix
# ---------------------------------------------------------------------------

def create_document_term_matrix(
    tokenized_docs: List[List[str]],
    method: str = "bow",
):
    """
    Build a document-term matrix for downstream modeling.

    Parameters
    ----------
    tokenized_docs : list of list of str
        Preprocessed, tokenized documents.
    method : str
        'bow'   — gensim Dictionary + corpus (for LDA)
        'tfidf' — sklearn TfidfVectorizer sparse matrix (for prediction)

    Returns
    -------
    tuple
        method='bow'   → (gensim.Dictionary, list of (id, count) corpus)
        method='tfidf' → (TfidfVectorizer, scipy sparse matrix)
    """
    if method == "bow":
        from gensim.corpora import Dictionary

        dictionary = Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=5, no_above=0.95)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        logger.info(
            "BoW corpus — vocab: %d tokens, %d documents",
            len(dictionary), len(corpus),
        )
        return dictionary, corpus

    elif method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts_flat = [" ".join(doc) for doc in tokenized_docs]
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
        )
        matrix = vectorizer.fit_transform(texts_flat)
        logger.info(
            "TF-IDF matrix — shape: %s, vocab: %d",
            matrix.shape, len(vectorizer.vocabulary_),
        )
        return vectorizer, matrix

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'bow' or 'tfidf'.")
