"""
test_pipeline.py
End-to-end smoke test for the Clinical NLP pipeline.
Tests: synthetic data generation, data loading, preprocessing, prediction, and fairness.

Run from project root:
    python tests/test_pipeline.py
"""

import sys
import traceback
import logging
from pathlib import Path

import numpy as np

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during tests

PASS = "  [PASS]"
FAIL = "  [FAIL]"
HEAD = "\n" + "-" * 60


def run(label: str, fn):
    """Run a test function, print result, return True if passed."""
    try:
        result = fn()
        print(f"{PASS}  {label}")
        if result is not None:
            print(f"         >> {result}")
        return True
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"         >> {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# SECTION 1 — Synthetic Data Generation
# ─────────────────────────────────────────────

def test_generate_patients():
    from src.generate_synthetic_data import generate_patients
    df = generate_patients(n=100)
    assert len(df) == 100
    assert set(["subject_id", "gender", "anchor_age"]).issubset(df.columns)
    assert df["anchor_age"].between(18, 90).all()
    return f"{len(df)} patients, age range {df['anchor_age'].min()}–{df['anchor_age'].max()}"


def test_generate_admissions():
    from src.generate_synthetic_data import generate_patients, generate_admissions
    patients = generate_patients(n=100)
    adm = generate_admissions(patients)
    assert len(adm) >= 100
    assert "readmission_30day" in adm.columns
    eligible = adm[adm["readmission_30day"] >= 0]
    rate = eligible["readmission_30day"].mean() * 100
    assert 5 < rate < 50, f"Readmission rate {rate:.1f}% looks off"
    return f"{len(adm)} admissions, readmit rate {rate:.1f}%"


def test_generate_notes():
    from src.generate_synthetic_data import generate_patients, generate_admissions, generate_discharge_notes
    patients = generate_patients(n=50)
    admissions = generate_admissions(patients)
    notes = generate_discharge_notes(patients, admissions)
    assert len(notes) == len(admissions)
    assert "text" in notes.columns
    assert notes["text"].str.len().min() > 100
    sample_text = notes["text"].iloc[0]
    assert "Chief Complaint" in sample_text
    assert "Brief Hospital Course" in sample_text
    return f"{len(notes)} notes, avg length {notes['text'].str.len().mean():.0f} chars"


def test_run_generates_csv_files():
    from src.generate_synthetic_data import run
    run(output_dir="data", n_patients=200)
    assert Path("data/synthetic_patients.csv").exists()
    assert Path("data/synthetic_admissions.csv").exists()
    assert Path("data/synthetic_discharge.csv").exists()
    return "All 3 CSV files written to data/"


# ─────────────────────────────────────────────
# SECTION 2 — Data Loader
# ─────────────────────────────────────────────

def _load_config():
    import yaml
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def test_load_discharge_notes():
    from src.data_loader import load_discharge_notes
    df = load_discharge_notes("data/synthetic_discharge.csv")
    assert len(df) > 0
    assert "text" in df.columns
    assert (df["note_type"] == "Discharge summary").all()
    return f"{len(df)} notes loaded"


def test_load_discharge_notes_sampling():
    from src.data_loader import load_discharge_notes
    df = load_discharge_notes("data/synthetic_discharge.csv", sample_size=50, random_seed=42)
    assert len(df) == 50
    return "Sampling to 50 notes works"


def test_load_admissions():
    from src.data_loader import load_admissions
    df = load_admissions("data/synthetic_admissions.csv")
    assert len(df) > 0
    assert "hadm_id" in df.columns
    import pandas as pd
    assert pd.api.types.is_datetime64_any_dtype(df["admittime"])
    return f"{len(df)} admissions, admittime parsed as datetime"


def test_load_patients():
    from src.data_loader import load_patients
    df = load_patients("data/synthetic_patients.csv")
    assert len(df) > 0
    assert "subject_id" in df.columns
    return f"{len(df)} patients"


def test_create_readmission_label():
    from src.data_loader import load_admissions, create_readmission_label
    adm = load_admissions("data/synthetic_admissions.csv")
    # Drop pre-computed label to test the function
    adm = adm.drop(columns=["readmission_30day"], errors="ignore")
    adm_labelled = create_readmission_label(adm, window_days=30)
    assert "readmission_30day" in adm_labelled.columns
    values = set(adm_labelled["readmission_30day"].unique())
    assert values.issubset({-1, 0, 1}), f"Unexpected label values: {values}"
    eligible = adm_labelled[adm_labelled["readmission_30day"] >= 0]
    rate = eligible["readmission_30day"].mean() * 100
    return f"Label values OK, readmit rate {rate:.1f}%"


def test_merge_dataset():
    from src.data_loader import load_discharge_notes, load_admissions, load_patients, merge_dataset
    notes = load_discharge_notes("data/synthetic_discharge.csv")
    adm   = load_admissions("data/synthetic_admissions.csv")
    pat   = load_patients("data/synthetic_patients.csv")
    merged = merge_dataset(notes, adm, pat)
    assert "los_days" in merged.columns
    assert "age_group" in merged.columns
    assert merged["los_days"].min() >= 0
    assert set(merged["age_group"].dropna().unique()).issubset({"<40", "40-65", "65+"})
    return f"Merged shape {merged.shape}, age groups: {merged['age_group'].value_counts().to_dict()}"


def test_load_all_convenience():
    from src.data_loader import load_all
    config = _load_config()
    df = load_all(config, use_synthetic=True)
    assert len(df) > 0
    assert "readmission_30day" in df.columns
    assert "age_group" in df.columns
    return f"load_all() shape={df.shape}"


def test_get_data_summary():
    from src.data_loader import load_all, get_data_summary
    config = _load_config()
    df = load_all(config, use_synthetic=True)
    summary = get_data_summary(df)
    assert "total_notes" in summary
    assert "readmission_rate_pct" in summary
    assert 5 < summary["readmission_rate_pct"] < 50
    return f"readmit rate {summary['readmission_rate_pct']}%, avg note {summary['avg_note_length_words']} words"


# ─────────────────────────────────────────────
# SECTION 3 — Preprocessing
# ─────────────────────────────────────────────

def test_clean_clinical_text():
    from src.preprocess import clean_clinical_text
    raw = "Patient ___ was admitted on 12/25/2022. WBC 12.3 mg, call (555) 123-4567."
    cleaned = clean_clinical_text(raw)
    assert "___" not in cleaned
    assert "12/25/2022" not in cleaned
    assert "(555) 123-4567" not in cleaned
    assert cleaned == cleaned.lower()
    return f"Cleaned: '{cleaned.strip()}'"


def test_clean_empty_text():
    from src.preprocess import clean_clinical_text
    assert clean_clinical_text("") == ""
    assert clean_clinical_text("   ") == ""
    assert clean_clinical_text(None) == ""
    return "Empty/None inputs handled safely"


def test_extract_sections():
    from src.preprocess import extract_sections
    note = """Some preamble text.

Chief Complaint:
Shortness of breath

History of Present Illness:
Patient presented with SOB.

Brief Hospital Course:
Patient improved with diuresis.
"""
    sections = extract_sections(note)
    assert "Chief Complaint" in sections
    assert "History of Present Illness" in sections
    assert "Brief Hospital Course" in sections
    assert "shortness" in sections["Chief Complaint"].lower()
    return f"Found sections: {list(sections.keys())}"


def test_remove_sections():
    from src.preprocess import remove_sections
    note = "Some text.\n\nDischarge Medications:\nListof meds here.\n\nBrief Hospital Course:\nCourse here."
    cleaned = remove_sections(note, ["Discharge Medications"])
    assert "Listof meds" not in cleaned
    assert "Course here" in cleaned
    return "Section removal OK"


def test_tokenize_clinical():
    from src.preprocess import tokenize_clinical
    text = "the patient was admitted for heart failure with fluid overload and shortness of breath"
    tokens = tokenize_clinical(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "patient" not in tokens   # in custom stopwords
    assert "the" not in tokens       # stopword
    return f"Tokens: {tokens}"


def test_tokenize_filters_numbers():
    from src.preprocess import tokenize_clinical
    text = "hemoglobin was 10.5 and sodium 138 with creatinine 2.1"
    tokens = tokenize_clinical(text)
    for tok in tokens:
        assert not tok.isdigit(), f"Pure digit token found: '{tok}'"
    return f"Filtered tokens: {tokens}"


def test_create_bigrams_trigrams():
    from src.preprocess import create_bigrams_trigrams
    # Use npmi scoring which works reliably on controlled repetitive corpora.
    # (default scoring needs large, varied real text — works on actual MIMIC notes)
    base_tokens = ["cardiac", "arrest", "pulmonary", "edema", "septic", "shock"]
    docs = [base_tokens * 5] * 80
    result = create_bigrams_trigrams(docs, min_count=3, threshold=0.5, scoring="npmi")
    assert len(result) == 80
    flat = [t for doc in result for t in doc]
    bigrams = [t for t in flat if "_" in t]
    assert len(bigrams) > 0, "Expected bigrams to form with npmi scoring"
    return f"Sample bigrams: {list(set(bigrams))[:4]}"


def test_full_pipeline():
    from src.data_loader import load_all
    from src.preprocess import build_preprocessing_pipeline
    config = _load_config()
    df = load_all(config, use_synthetic=True).head(30)
    processed = build_preprocessing_pipeline(
        df, config=config, use_scispacy=False, use_phrases=False
    )
    assert "cleaned_text" in processed.columns
    assert "tokens" in processed.columns
    assert "num_tokens" in processed.columns
    assert processed["num_tokens"].min() > 0
    return f"30 docs processed, avg tokens {processed['num_tokens'].mean():.1f}"


def test_bow_corpus():
    from src.data_loader import load_all
    from src.preprocess import build_preprocessing_pipeline, create_document_term_matrix
    config = _load_config()
    df = load_all(config, use_synthetic=True).head(30)
    processed = build_preprocessing_pipeline(df, config=config, use_phrases=False)
    dictionary, corpus = create_document_term_matrix(processed["tokens"].tolist(), method="bow")
    assert len(corpus) == len(processed)
    assert len(dictionary) > 0
    return f"BoW — vocab {len(dictionary)}, docs {len(corpus)}"


def test_tfidf_matrix():
    from src.data_loader import load_all
    from src.preprocess import build_preprocessing_pipeline, create_document_term_matrix
    config = _load_config()
    df = load_all(config, use_synthetic=True).head(30)
    processed = build_preprocessing_pipeline(df, config=config, use_phrases=False)
    vectorizer, matrix = create_document_term_matrix(processed["tokens"].tolist(), method="tfidf")
    assert matrix.shape[0] == len(processed)
    assert matrix.shape[1] > 0
    return f"TF-IDF — shape {matrix.shape}"


# ─────────────────────────────────────────────
# SECTION 4 — Prediction Models
# ─────────────────────────────────────────────

def _build_small_feature_sets():
    """Helper: build feature sets from a small synthetic dataset for testing."""
    from src.data_loader import load_all
    from src.preprocess import build_preprocessing_pipeline
    from src.feature_engineer import build_feature_sets
    config = _load_config()
    df = load_all(config, use_synthetic=True).head(200)
    processed = build_preprocessing_pipeline(df, config=config, use_scispacy=False, use_phrases=False)
    return build_feature_sets(processed, config=config)


def test_split_data():
    from src.predict import split_data
    X = np.random.randn(100, 5)
    y = np.array([0] * 70 + [1] * 30)
    splits = split_data(X, y, test_size=0.2, val_size=0.1)
    assert splits["X_train"].shape[0] == 70
    assert splits["X_val"].shape[0] == 10
    assert splits["X_test"].shape[0] == 20
    return f"train={splits['X_train'].shape[0]}, val={splits['X_val'].shape[0]}, test={splits['X_test'].shape[0]}"


def test_get_model():
    from src.predict import get_model
    loaded = []
    for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        try:
            model = get_model(name)
            assert hasattr(model, "fit")
            assert hasattr(model, "predict_proba")
            loaded.append(name)
        except (ImportError, ModuleNotFoundError):
            pass  # optional dependency not installed
    assert len(loaded) >= 2, f"At least 2 models should be available, got: {loaded}"
    return f"Instantiated: {loaded}"


def test_train_and_evaluate():
    from src.predict import train_model, evaluate_model
    X = np.random.randn(200, 10)
    y = np.array([0] * 140 + [1] * 60)
    np.random.shuffle(y)

    model = train_model("logistic_regression", X[:160], y[:160])
    metrics = evaluate_model(model, X[160:], y[160:], model_name="logistic_regression")
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["roc_auc"] <= 1
    return f"LR test — F1: {metrics['f1']}, ROC-AUC: {metrics['roc_auc']}"


def test_cross_validate():
    from src.predict import cross_validate_model
    X = np.random.randn(150, 10)
    y = np.array([0] * 100 + [1] * 50)
    result = cross_validate_model("logistic_regression", X, y, n_folds=3)
    assert result["n_folds"] == 3
    assert 0 <= result["roc_auc_mean"] <= 1
    return f"CV — F1: {result['f1_mean']}±{result['f1_std']}, ROC-AUC: {result['roc_auc_mean']}±{result['roc_auc_std']}"


def test_feature_importance():
    from src.predict import get_feature_importance
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.randn(100, 5)
    y = np.array([0] * 70 + [1] * 30)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    imp = get_feature_importance(model, ["f0", "f1", "f2", "f3", "f4"], top_n=3)
    assert len(imp) == 3
    assert "importance" in imp.columns
    return f"Top feature: {imp.iloc[0]['feature']} (imp={imp.iloc[0]['importance']:.4f})"


def test_optimal_threshold():
    from src.predict import find_optimal_threshold
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([np.random.uniform(0, 0.6, 50), np.random.uniform(0.4, 1.0, 50)])
    thresh, score = find_optimal_threshold(y_true, y_prob, metric="f1")
    assert 0.1 <= thresh <= 0.9
    assert 0 <= score <= 1
    return f"Optimal threshold: {thresh}, F1: {score}"


def test_full_prediction_pipeline():
    feature_sets = _build_small_feature_sets()
    from src.predict import run_prediction_pipeline
    config = _load_config()
    # Only test 2 models and 2 feature types to keep it fast
    config["prediction"]["models"] = ["logistic_regression", "random_forest"]
    config["prediction"]["feature_types"] = ["tfidf", "structured"]
    result = run_prediction_pipeline(feature_sets, config=config)
    n_results = len(result["results"])
    assert n_results > 0, "Expected at least some results"
    assert result["results_df"] is not None
    best = result["best"]
    return f"Best: {best.get('model')} + {best.get('feature_type')} — ROC-AUC: {best.get('roc_auc')}"


# ─────────────────────────────────────────────
# SECTION 5 — Fairness Analysis
# ─────────────────────────────────────────────

def test_group_metrics():
    import pandas as pd
    from src.fairness import compute_group_metrics
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.2, 0.8, 0.3, 0.4, 0.1, 0.9, 0.6, 0.2, 0.7, 0.3])
    sf = pd.Series(["M", "M", "F", "F", "M", "M", "F", "F", "M", "F"])
    df = compute_group_metrics(y_true, y_pred, y_prob, sf, attribute_name="gender")
    assert len(df) == 2
    assert "accuracy" in df.columns
    assert "fpr" in df.columns
    return f"Groups: {list(df.index)}, cols: {list(df.columns)}"


def test_fairness_metrics():
    import pandas as pd
    from src.fairness import compute_fairness_metrics
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0])
    sf = pd.Series(["M", "M", "F", "F", "M", "M", "F", "F", "M", "F"])
    result = compute_fairness_metrics(y_true, y_pred, sf, attribute_name="gender")
    assert "demographic_parity_difference" in result
    assert "equalized_odds_difference" in result
    assert "fpr_difference" in result
    return f"DPD: {result['demographic_parity_difference']}, EOD: {result['equalized_odds_difference']}"


def test_fairness_audit():
    import pandas as pd
    from src.fairness import run_fairness_audit
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.3, n)
    y_prob = np.clip(y_true * 0.6 + np.random.uniform(0, 0.4, n), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    protected_df = pd.DataFrame({
        "gender": np.random.choice(["M", "F"], n),
        "insurance": np.random.choice(["Medicare", "Medicaid", "Other"], n),
    })
    result = run_fairness_audit(y_true, y_pred, y_prob, protected_df)
    assert "group_metrics" in result
    assert "summary_df" in result
    assert len(result["group_metrics"]) == 2
    return f"Audited {len(result['group_metrics'])} attributes, flags reported"


# ─────────────────────────────────────────────
# SECTION 5.5 — Embeddings
# ─────────────────────────────────────────────

def test_load_embedding_model():
    from src.embeddings import load_embedding_model
    tokenizer, model, device = load_embedding_model("clinicalbert", device="cpu")
    assert tokenizer is not None
    assert model is not None
    assert model.config.hidden_size == 768
    return f"ClinicalBERT loaded, hidden_size={model.config.hidden_size}, device={device}"


def test_embed_single_texts():
    from src.embeddings import embed_texts
    texts = [
        "Patient admitted for acute heart failure with reduced ejection fraction.",
        "Presented with community-acquired pneumonia and right lower lobe infiltrate.",
        "",  # empty text should produce a zero vector
    ]
    emb, meta = embed_texts(texts, model_name="clinicalbert", device="cpu", show_progress=False)
    assert emb.shape == (3, 768), f"Expected (3, 768), got {emb.shape}"
    assert meta["n_documents"] == 3
    # Non-empty texts should have non-zero embeddings
    assert np.linalg.norm(emb[0]) > 0, "First embedding should be non-zero"
    assert np.linalg.norm(emb[1]) > 0, "Second embedding should be non-zero"
    # Empty text should be zero vector
    assert np.linalg.norm(emb[2]) == 0.0, "Empty text should produce zero vector"
    return f"Shape: {emb.shape}, chunked: {meta['n_chunked']}/{meta['n_documents']}"


def test_embed_long_text_chunking():
    from src.embeddings import embed_texts
    # Create a text long enough to require chunking (>510 tokens)
    long_text = "The patient presented with " + " ".join(["clinical finding"] * 300)
    emb, meta = embed_texts([long_text], model_name="clinicalbert", device="cpu", show_progress=False)
    assert emb.shape == (1, 768)
    assert meta["n_chunked"] == 1, "Long text should have been chunked"
    assert np.linalg.norm(emb[0]) > 0
    return f"Chunked 1 long doc, embedding norm: {np.linalg.norm(emb[0]):.4f}"


def test_reduce_embeddings():
    from src.embeddings import reduce_embeddings
    fake_emb = np.random.randn(50, 768).astype(np.float32)
    reduced, reducer = reduce_embeddings(fake_emb, n_components=20, method="pca")
    assert reduced.shape == (50, 20), f"Expected (50, 20), got {reduced.shape}"
    assert reducer is not None
    return f"PCA: {fake_emb.shape} -> {reduced.shape}"


# ─────────────────────────────────────────────
# SECTION 6 — Visualization
# ─────────────────────────────────────────────

def test_plot_demographics():
    from src.data_loader import load_all
    from src.visualize import plot_demographics
    config = _load_config()
    df = load_all(config, use_synthetic=True)
    paths = plot_demographics(df, config)
    assert len(paths) >= 1
    assert all(Path(p).exists() for p in paths)
    return f"Saved {len(paths)} file(s): {[Path(p).name for p in paths]}"


def test_plot_note_length():
    from src.data_loader import load_all
    from src.visualize import plot_note_length_distribution
    config = _load_config()
    df = load_all(config, use_synthetic=True)
    path = plot_note_length_distribution(df, config)
    assert Path(path).exists()
    return f"Saved: {Path(path).name}"


def test_plot_roc_pr_confusion():
    from src.visualize import plot_roc_curves, plot_pr_curves, plot_confusion_matrices
    config = _load_config()
    # Build minimal mock results
    fpr = np.array([0, 0.2, 0.5, 1.0])
    tpr = np.array([0, 0.6, 0.8, 1.0])
    prec = np.array([1.0, 0.7, 0.5, 0.3])
    rec = np.array([0.0, 0.4, 0.7, 1.0])
    mock_results = [{
        "model": "logistic_regression", "feature_type": "tfidf",
        "roc_auc": 0.75, "pr_auc": 0.60,
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "pr_curve": {"precision": prec, "recall": rec},
        "confusion_matrix": [[40, 10], [5, 15]],
    }]
    p1 = plot_roc_curves(mock_results, config)
    p2 = plot_pr_curves(mock_results, config)
    p3 = plot_confusion_matrices(mock_results, config)
    assert Path(p1).exists() and Path(p2).exists() and Path(p3).exists()
    return f"Saved: {Path(p1).name}, {Path(p2).name}, {Path(p3).name}"


def test_plot_fairness():
    import pandas as pd
    from src.visualize import plot_fairness_disparity, plot_fairness_group_metrics
    config = _load_config()
    summary_df = pd.DataFrame({
        "demographic_parity_difference": [0.05, 0.12],
        "equalized_odds_difference": [0.08, 0.15],
    }, index=["gender", "insurance"])
    p1 = plot_fairness_disparity(summary_df, config)
    assert Path(p1).exists()

    group_df = pd.DataFrame({
        "accuracy": [0.80, 0.75],
        "precision": [0.70, 0.65],
        "recall": [0.60, 0.55],
        "f1": [0.65, 0.60],
        "fpr": [0.15, 0.20],
    }, index=["M", "F"])
    p2 = plot_fairness_group_metrics({"gender": group_df}, config)
    assert Path(p2).exists()
    return f"Saved: {Path(p1).name}, {Path(p2).name}"


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def main():
    results = []

    print(HEAD)
    print("  SECTION 1 — Synthetic Data Generation")
    print("-" * 60)
    results.append(run("generate_patients()", test_generate_patients))
    results.append(run("generate_admissions()", test_generate_admissions))
    results.append(run("generate_discharge_notes()", test_generate_notes))
    results.append(run("run() writes CSV files", test_run_generates_csv_files))

    print(HEAD)
    print("  SECTION 2 — Data Loader")
    print("-" * 60)
    results.append(run("load_discharge_notes()", test_load_discharge_notes))
    results.append(run("load_discharge_notes() sampling", test_load_discharge_notes_sampling))
    results.append(run("load_admissions()", test_load_admissions))
    results.append(run("load_patients()", test_load_patients))
    results.append(run("create_readmission_label()", test_create_readmission_label))
    results.append(run("merge_dataset()", test_merge_dataset))
    results.append(run("load_all() convenience", test_load_all_convenience))
    results.append(run("get_data_summary()", test_get_data_summary))

    print(HEAD)
    print("  SECTION 3 — Preprocessing")
    print("-" * 60)
    results.append(run("clean_clinical_text()", test_clean_clinical_text))
    results.append(run("clean_clinical_text() edge cases", test_clean_empty_text))
    results.append(run("extract_sections()", test_extract_sections))
    results.append(run("remove_sections()", test_remove_sections))
    results.append(run("tokenize_clinical()", test_tokenize_clinical))
    results.append(run("tokenize_clinical() filters numbers", test_tokenize_filters_numbers))
    results.append(run("create_bigrams_trigrams()", test_create_bigrams_trigrams))
    results.append(run("build_preprocessing_pipeline()", test_full_pipeline))
    results.append(run("create_document_term_matrix() BoW", test_bow_corpus))
    results.append(run("create_document_term_matrix() TF-IDF", test_tfidf_matrix))

    print(HEAD)
    print("  SECTION 4 — Prediction Models")
    print("-" * 60)
    results.append(run("split_data()", test_split_data))
    results.append(run("get_model() all types", test_get_model))
    results.append(run("train + evaluate (LR)", test_train_and_evaluate))
    results.append(run("cross_validate_model()", test_cross_validate))
    results.append(run("get_feature_importance()", test_feature_importance))
    results.append(run("find_optimal_threshold()", test_optimal_threshold))
    results.append(run("run_prediction_pipeline()", test_full_prediction_pipeline))

    print(HEAD)
    print("  SECTION 5 — Fairness Analysis")
    print("-" * 60)
    results.append(run("compute_group_metrics()", test_group_metrics))
    results.append(run("compute_fairness_metrics()", test_fairness_metrics))
    results.append(run("run_fairness_audit()", test_fairness_audit))

    print(HEAD)
    print("  SECTION 5.5 — Embeddings")
    print("-" * 60)
    results.append(run("load_embedding_model()", test_load_embedding_model))
    results.append(run("embed_texts() basic", test_embed_single_texts))
    results.append(run("embed_texts() long text chunking", test_embed_long_text_chunking))
    results.append(run("reduce_embeddings() PCA", test_reduce_embeddings))

    print(HEAD)
    print("  SECTION 6 — Visualization")
    print("-" * 60)
    results.append(run("plot_demographics()", test_plot_demographics))
    results.append(run("plot_note_length_distribution()", test_plot_note_length))
    results.append(run("plot_roc / pr / confusion", test_plot_roc_pr_confusion))
    results.append(run("plot_fairness()", test_plot_fairness))

    passed = sum(results)
    total = len(results)
    print(HEAD)
    print(f"  RESULTS: {passed}/{total} tests passed")
    print("-" * 60)
    if passed == total:
        print("  All tests passed. Pipeline is healthy.")
    else:
        print(f"  {total - passed} test(s) failed — see above for details.")
    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
