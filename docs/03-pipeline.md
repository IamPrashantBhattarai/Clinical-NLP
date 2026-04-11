# 3. Pipeline Reference

Module-by-module reference for everything under [src/](../src/). Functions are listed in the order they appear in each file; names link to the source.

---

## 3.1 `src/generate_synthetic_data.py`

Generates a MIMIC-IV-style cohort locally — no credentialed access required. Writes three CSVs under `data/`:

- `synthetic_discharge.csv` — discharge summaries with realistic clinical sections
- `synthetic_admissions.csv` — admission/discharge timestamps + insurance + admission_type
- `synthetic_patients.csv` — demographics

Entry point: `run(output_dir="data", n_patients=200)` (imported in the notebook as `generate_data`). Uses `random` + `numpy` with a fixed seed (42) so outputs are reproducible.

Text generation builds each note from category-specific templates (chief complaints, past medical history, meds, social history, hospital course, discharge plan) so LDA/BERTopic actually find meaningful topics at small scale.

---

## 3.2 `src/data_loader.py`

Loads raw tables and produces the merged cohort used everywhere downstream.

| Function | Purpose |
|---|---|
| [load_discharge_notes](../src/data_loader.py#L20) | Read CSV, filter to `note_type == "Discharge summary"`, optional subsample. |
| [load_admissions](../src/data_loader.py#L68) | Parse admission/discharge timestamps, keep utilization fields. |
| [load_patients](../src/data_loader.py#L97) | Parse demographics; derives `age_group`. |
| [create_readmission_label](../src/data_loader.py#L125) | Compute binary 30-day readmission from consecutive admissions per `subject_id`. |
| [merge_dataset](../src/data_loader.py#L190) | Join notes + admissions + patients, produce `los_days`, `num_prior_admissions`, etc. |
| [get_data_summary](../src/data_loader.py#L243) | Print a compact human-readable summary. |
| [load_all](../src/data_loader.py#L329) | Top-level helper: picks synthetic vs real paths from config and returns the merged frame. |

**Readmission label semantics.** `-1` = ineligible (e.g., expired, last admission for the patient, insufficient follow-up). Downstream code filters with `processed["readmission_30day"] >= 0` before LDA/prediction.

---

## 3.3 `src/preprocess.py`

PHI-aware cleaning, tokenization, and optional scispaCy.

| Function | Purpose |
|---|---|
| [clean_clinical_text](../src/preprocess.py#L101) | Remove MIMIC PHI patterns (`[**...**]`), dates, excess whitespace, punctuation. |
| [extract_sections](../src/preprocess.py#L143) | Parse section headers (e.g., `History of Present Illness:`) into a dict. |
| [remove_sections](../src/preprocess.py#L179) | Drop sections listed in config (`Discharge Medications`, etc.) that leak the target. |
| [tokenize_clinical](../src/preprocess.py#L242) | Tokenize via spaCy (if available) or NLTK, strip clinical stopwords, lemmatize. |
| [create_bigrams_trigrams](../src/preprocess.py#L301) | Gensim `Phrases` for medical multi-word terms (`heart_failure`, `atrial_fibrillation`). |
| [build_preprocessing_pipeline](../src/preprocess.py#L355) | End-to-end: clean → section-removal → tokenize → phrases → length filter. Adds `cleaned_text`, `tokens`, `num_tokens` columns. |
| [create_document_term_matrix](../src/preprocess.py#L466) | Helper for building a Gensim dictionary+corpus used by LDA. |

### Stopwords

A union of NLTK English stopwords, a medical-noise list (`patient`, `pt`, `mg`, `ml`, `hospital`, `admission`, `discharge`, …), and any custom list from `preprocessing.custom_stopwords` in config.

### Lazy heavy imports

spaCy and NLTK are loaded on first call via `_get_spacy()` and `_ensure_nltk()` so `import src.preprocess` stays cheap for unit tests.

---

## 3.4 `src/embeddings.py`

Dense clinical-text embeddings via Hugging Face transformers.

| Function | Purpose |
|---|---|
| [load_embedding_model](../src/embeddings.py#L35) | Load a model + tokenizer from the `MODEL_REGISTRY` shortcut (`clinicalbert`, `biobert`, `pubmedbert`, `bert`) or any HF model ID. Auto-selects CUDA. |
| [_embed_single_text](../src/embeddings.py#L78) | Split a long note into overlapping chunks (`max_length=512`, `stride=256`), mean/CLS-pool each chunk, average. |
| [embed_texts](../src/embeddings.py#L177) | Batch embedder with a tqdm progress bar; returns `(n_docs, d)` float32 matrix + model metadata. |
| [reduce_embeddings](../src/embeddings.py#L260) | Optional PCA (default) or UMAP compression to `n_components`. |

**Why chunking.** Discharge notes routinely exceed BERT's 512-token limit. The function slides a window with configurable stride and averages chunk vectors — a common practice for long clinical documents.

**Defaults** (from `config/config.yaml → embeddings`): `clinicalbert`, mean pooling, max_length=512, stride=256, PCA to 50 components.

---

## 3.5 `src/topic_model.py`

Two complementary topic models.

### LDA

| Function | Purpose |
|---|---|
| [train_lda](../src/topic_model.py#L86) | Fit Gensim `LdaModel` with configurable `num_topics`, `passes`, `iterations`, `alpha`, `eta`. |
| [evaluate_coherence](../src/topic_model.py#L173) | `CoherenceModel` (`c_v`) over a fitted LDA. |
| [find_optimal_topics](../src/topic_model.py#L221) | Grid-search topic counts (from `lda.num_topics` list), pick the one with highest coherence. |
| [get_topic_words](../src/topic_model.py#L275) | Extract top-N words per topic as `{tid: [(word, weight), …]}`. |
| [label_topics](../src/topic_model.py#L304) | Score each topic against `CLINICAL_LABEL_KEYWORDS` (Cardiovascular, Respiratory, Sepsis, Renal, Neurological, …) and assign the best-matching label. |
| [get_document_topics](../src/topic_model.py#L356) | Full doc-topic probability matrix `(n_docs, n_topics)`. |
| [topics_to_dataframe](../src/topic_model.py#L390) | Pretty summary table for notebooks. |
| [run_lda_pipeline](../src/topic_model.py#L683) | One-shot wrapper: build corpus → grid-search → label → return `lda_results` dict. |

### BERTopic

| Function | Purpose |
|---|---|
| [train_bertopic](../src/topic_model.py#L428) | BioBERT (`dmis-lab/biobert-base-cased-v1.2`) + UMAP + HDBSCAN, with a `sentence-transformers/all-MiniLM-L6-v2` fallback. Auto-scales HDBSCAN/UMAP parameters for small datasets (`n < 500`) and disables `nr_topics='auto'` in that regime. |
| [compare_models](../src/topic_model.py#L535) | Compute coherence and topic diversity for LDA and BERTopic side-by-side. |
| [test_topic_readmission_association](../src/topic_model.py#L622) | Mann-Whitney U test: for each topic, compare topic-probability distributions between readmitted vs non-readmitted patients. Reports per-topic p-values. |

### `lda_results` schema

```python
{
  "best_model":         LdaModel,
  "best_num_topics":    int,
  "coherence_scores":   {num_topics: c_v},
  "topic_words":        {tid: [(word, weight), ...]},
  "topic_labels":       {tid: "Cardiovascular", ...},
  "doc_topic_matrix":   np.ndarray,   # (n_docs, n_topics)
  "dictionary":         gensim.corpora.Dictionary,
  "corpus":             [[(token_id, count), ...], ...],
  "topics_df":          pd.DataFrame,
}
```

---

## 3.6 `src/feature_engineer.py`

Builds the feature dict consumed by the prediction pipeline.

| Function | Purpose |
|---|---|
| [create_tfidf_features](../src/feature_engineer.py#L28) | Fits/transforms a `TfidfVectorizer` with `sublinear_tf=True` (helps with long notes). |
| [create_topic_features](../src/feature_engineer.py#L90) | Document-topic probabilities for LDA **or** BERTopic. Returns `(n_docs, n_topics)` dense float32. Handles 1-D `probs` from BERTopic by one-hot encoding topic assignments. |
| [create_structured_features](../src/feature_engineer.py#L147) | Age (z-scored), gender, insurance/admission-type one-hots, `log1p(los_days)`, `num_prior_admissions`, `days_since_last_admission`, `prior_expire_flag`. |
| [create_text_statistics_features](../src/feature_engineer.py#L260) | Note length, token count, sentence count, negation count, section count, readability proxies. |
| [combine_features](../src/feature_engineer.py#L348) | Sparse-aware horizontal stack (`scipy.sparse.hstack`) preserving name vectors. |
| [build_feature_sets](../src/feature_engineer.py#L410) | Top-level: takes `processed_df`, optional `lda_results` / `bertopic_model` / `embedding_matrix`, returns the full `feature_sets` dict used by `run_prediction_pipeline`. |

### `feature_sets` schema

```python
{
  "tfidf":          {"X": sparse, "vectorizer": TfidfVectorizer, "names": [...]},
  "topic_lda":      {"X": ndarray, "names": [...]} | None,
  "topic_bertopic": {"X": ndarray, "names": [...]} | None,
  "embeddings":     {"X": ndarray, "names": [...]} | None,
  "structured":     {"X": ndarray, "names": [...]},
  "text_stats":     {"X": ndarray, "names": [...]},
  "combined":       {"X": sparse,  "names": [...]},
  "label":          ndarray,          # y (readmission_30day, int)
  "vectorizer":     TfidfVectorizer,  # duplicate for convenience
}
```

**Eligibility alignment.** `build_feature_sets` slices the processed frame with the `readmission_30day >= 0` mask before building feature matrices — any external inputs (LDA corpus, embedding matrix) are auto-realigned if their row counts differ from the eligible subset.

---

## 3.7 `src/feature_selection.py`

Applied inside `run_prediction_pipeline` when `prediction.feature_selection.enabled = True`. Each function returns `(indices, selected_names)`; `apply_selection` slices a matrix by indices so train/val/test stay aligned.

| Function | Method |
|---|---|
| [variance_threshold_selection](../src/feature_selection.py#L29) | Drop (near-)constant features via `VarianceThreshold`. |
| [univariate_selection](../src/feature_selection.py#L61) | `SelectKBest` with `chi2`, `f_classif`, or `mutual_info_classif`. |
| [l1_selection](../src/feature_selection.py#L108) | Fit an L1 Logistic Regression; keep non-zero coefficients. |
| [rfe_selection](../src/feature_selection.py#L155) | `sklearn.feature_selection.RFE` with a LogReg base estimator. |
| [shap_selection](../src/feature_selection.py#L206) | Rank by mean \|SHAP\|, keep `top_k`. Trains a lightweight model on a subsample for the scoring. |
| [apply_selection](../src/feature_selection.py#L252) | Slice a feature matrix by an index array (sparse-safe). |
| [select_features](../src/feature_selection.py#L263) | Dispatcher that calls one of the above by `method` name and returns `{X_selected, indices, names, n_before, n_after, method}`. |

Only feature sets listed in `prediction.feature_selection.apply_to` (default `["tfidf", "combined"]`) are reduced — topic/structured/text_stats sets are already low-dimensional.

---

## 3.8 `src/predict.py`

Model training, tuning, evaluation, and the top-level pipeline runner.

| Function | Purpose |
|---|---|
| [split_data](../src/predict.py#L40) | Stratified train/val/test split with configurable sizes. |
| [apply_smote](../src/predict.py#L89) | SMOTE oversampling on training data (optional, for imbalanced classes). |
| [get_model](../src/predict.py#L118) | Factory for `logistic_regression`, `random_forest`, `xgboost`, `lightgbm` with class-balanced defaults. |
| [tune_hyperparameters](../src/predict.py#L219) | `RandomizedSearchCV` with stratified k-fold; returns best estimator + best params + CV score. |
| [train_model](../src/predict.py#L297) | Fit a model on `(X_train, y_train)` with optional SMOTE. |
| [evaluate_model](../src/predict.py#L363) | Accuracy / precision / recall / F1 / ROC-AUC / PR-AUC + confusion matrix. |
| [cross_validate_model](../src/predict.py#L423) | Stratified k-fold CV for the final selected model. |
| [get_feature_importance](../src/predict.py#L480) | Pulls `coef_` or `feature_importances_` into a named DataFrame. |
| [save_models](../src/predict.py#L514) | Persist every trained model + mark the best one (`*_BEST.joblib`). |
| [load_model](../src/predict.py#L555) | Load a joblib file. |
| [find_optimal_threshold](../src/predict.py#L564) | Pick a probability cutoff that maximizes F1 on the validation set. |
| [run_prediction_pipeline](../src/predict.py#L607) | Top-level: for each `(feature_type, model)` combo, split → optional feature selection → optional tuning → train → evaluate. Returns results, CV scores, best model, tuned params, feature-selection summary, trained models, splits. |

### `prediction_results` schema

```python
{
  "results":           [dict, ...],          # rows of metrics
  "results_df":        pd.DataFrame,
  "best":              {"model", "feature_type", ... metrics ...},
  "models":            {(model, feat): fitted_estimator},
  "splits":            {feat: {"X_train", "X_val", "X_test", "y_*", "feature_names"}},
  "cv_results":        [dict, ...],
  "feature_importances": {(model, feat): DataFrame},
  "tuned_params":      {(model, feat): {"best_params", "cv_score"}},
  "feature_selection": {feat: {"method", "n_before", "n_after", "indices", "names"}},
  "saved_model_paths": {(model, feat): Path},
}
```

---

## 3.9 `src/explainability.py`

SHAP wrappers that pick the right explainer for the model family.

| Function | Purpose |
|---|---|
| [get_explainer](../src/explainability.py#L33) | `TreeExplainer` for RF/XGBoost/LightGBM; `LinearExplainer` for Logistic Regression. |
| [compute_shap_values](../src/explainability.py#L78) | Run the explainer on `X_explain`, return a `shap.Explanation` with a values matrix and base value. |
| [shap_global_importance](../src/explainability.py#L159) | Mean \|SHAP\| per feature → ranked DataFrame. |
| [explain_patient](../src/explainability.py#L193) | Per-row top-N contributors sorted by absolute impact. |
| [run_shap_analysis](../src/explainability.py#L253) | Orchestrates the above for the best model from `prediction_results`; returns `shap_results` dict consumed by `visualize.py` and the notebook. |

---

## 3.10 `src/fairness.py`

Group-level auditing with Fairlearn.

| Function | Purpose |
|---|---|
| [compute_group_metrics](../src/fairness.py#L41) | Per-group: accuracy, precision, recall, F1, FPR, FNR, selection rate (via `MetricFrame`). |
| [compute_fairness_metrics](../src/fairness.py#L100) | Disparity: `demographic_parity_difference`, `equalized_odds_difference`, `fpr_difference`, `fnr_difference`. |
| [run_fairness_audit](../src/fairness.py#L156) | Loop over every protected attribute listed in `config.fairness.protected_attributes` and run both. |

### `fairness_results` schema

```python
{
  "group_metrics":    {attr: DataFrame},  # per-group metrics table
  "fairness_metrics": {attr: {metric: value}},
  "summary_df":       DataFrame,          # compact human-readable table
}
```

---

## 3.11 `src/visualize.py`

Matplotlib+Seaborn plotting with a consistent style. Every `plot_*` function returns the `Figure` and writes a PNG under `results/figures/` via `_save_fig`.

| Function | Output PNG |
|---|---|
| [plot_demographics](../src/visualize.py#L65) | `demographics.png` |
| [plot_note_length_distribution](../src/visualize.py#L132) | `note_length_distribution.png` |
| [plot_coherence_scores](../src/visualize.py#L171) | `coherence_scores.png` |
| [plot_topic_word_clouds](../src/visualize.py#L194) | `topic_word_clouds.png` |
| [plot_topic_readmission_heatmap](../src/visualize.py#L238) | `topic_readmission_heatmap.png` |
| [plot_roc_curves](../src/visualize.py#L268) | `roc_curves.png` |
| [plot_pr_curves](../src/visualize.py#L303) | `pr_curves.png` |
| [plot_confusion_matrices](../src/visualize.py#L336) | `confusion_matrices.png` |
| [plot_model_comparison](../src/visualize.py#L371) | `model_comparison.png` |
| [plot_feature_importance](../src/visualize.py#L399) | `feature_importance.png` |
| [plot_fairness_disparity](../src/visualize.py#L438) | `fairness_disparity.png` |
| [plot_fairness_group_metrics](../src/visualize.py#L474) | `fairness_group_metrics.png` |
| [plot_shap_global_importance](../src/visualize.py#L510) | `shap_global_importance.png` |
| [plot_shap_summary](../src/visualize.py#L551) | `shap_summary.png` |
| [plot_shap_patient_explanation](../src/visualize.py#L597) | `shap_patient_*_top*.png` |
| [generate_all_figures](../src/visualize.py#L651) | Orchestrator — one call to produce everything. |

The backend uses `matplotlib.use("Agg")` so figures render in headless runs (CI, Docker).

---

## 3.12 `src/exporter.py`

Serializes in-memory pipeline outputs into the three JSON files the dashboard reads.

| Function | Output |
|---|---|
| [build_results_payload](../src/exporter.py#L58) | Model comparison rows + best model → `results.json`. |
| [build_fairness_payload](../src/exporter.py#L97) | Per-attribute groups + disparities → `fairness.json`. |
| [build_topics_payload](../src/exporter.py#L133) | LDA topics, words, per-topic readmission rate → `topics.json`. |
| [export_dashboard_json](../src/exporter.py#L194) | Top-level entry — call from a notebook or script after the pipeline finishes. Any missing input skips its file (dashboard falls back to mock data). |

The `results/exports/` directory is the single contract between the training pipeline and the dashboard. See [docs/05-dashboard.md](05-dashboard.md).
