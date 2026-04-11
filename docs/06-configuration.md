# 6. Configuration Reference

Everything the pipeline reads from disk is centralized in [config/config.yaml](../config/config.yaml). This page lists every top-level section, its keys, defaults, and what changing them does.

The config is loaded once in the notebook and passed as a plain `dict` to every module — no magic, no env vars.

```python
import yaml
config = yaml.safe_load(open("config/config.yaml"))
```

---

## 6.1 `data:`

Paths to the raw tables and global sampling knobs.

```yaml
data:
  mimic_notes_path:          "data/discharge.csv"
  mimic_admissions_path:     "data/admissions.csv"
  mimic_patients_path:       "data/patients.csv"
  synthetic_notes_path:      "data/synthetic_discharge.csv"
  synthetic_admissions_path: "data/synthetic_admissions.csv"
  synthetic_patients_path:   "data/synthetic_patients.csv"
  sample_size:               50000
  random_seed:               42
```

| Key | What it does |
|---|---|
| `mimic_*_path` | Paths used when `load_all(use_synthetic=False)`. Point at your real MIMIC-IV CSVs. |
| `synthetic_*_path` | Paths used when `use_synthetic=True`. The generator writes to these. |
| `sample_size` | If set, `load_discharge_notes` randomly samples this many notes. Set `null` (or remove) to disable. |
| `random_seed` | Global seed for sampling and splits. |

---

## 6.2 `preprocessing:`

```yaml
preprocessing:
  min_note_length: 100
  max_note_length: 50000
  remove_sections:
    - "Discharge Medications"
    - "Discharge Disposition"
    - "Discharge Diagnosis"
    - "Discharge Condition"
  custom_stopwords:
    - "patient"
    - "pt"
    - "history"
    - "mg"
    - "ml"
    - "dr"
    - "hospital"
    - "admission"
    - "discharge"
    - "date"
```

| Key | What it does |
|---|---|
| `min_note_length` | Drops notes shorter than this many characters (too short to be useful). |
| `max_note_length` | Truncates extremely long notes before tokenization to bound memory. |
| `remove_sections` | Headers whose bodies are removed by `remove_sections` in preprocess. Default list strips discharge-planning sections that leak the outcome. |
| `custom_stopwords` | Domain-specific tokens to drop on top of NLTK English stopwords. |

---

## 6.3 `topic_modeling:`

Two independent sub-blocks for LDA and BERTopic.

### 6.3.1 `topic_modeling.lda`

```yaml
lda:
  num_topics: [5, 10, 15, 20]
  passes: 15
  iterations: 100
  chunksize: 2000
  eval_every: null
  alpha: "auto"
  eta: "auto"
  best_num_topics: null
```

| Key | What it does |
|---|---|
| `num_topics` | Grid of topic counts to search. `find_optimal_topics` picks the one with the highest `c_v` coherence. |
| `passes` | Full passes over the corpus during training. |
| `iterations` | Max iterations per document during inference. |
| `chunksize` | Number of documents processed at a time. Bigger = faster but more memory. |
| `eval_every` | Gensim perplexity logging interval (`null` disables). |
| `alpha`, `eta` | Dirichlet priors. `"auto"` learns them. |
| `best_num_topics` | If set, skips the coherence grid and trains only this topic count. |

### 6.3.2 `topic_modeling.bertopic`

```yaml
bertopic:
  embedding_model: "dmis-lab/biobert-base-cased-v1.2"
  min_topic_size: 50
  nr_topics: "auto"
  top_n_words: 15
  umap_n_neighbors: 15
  umap_n_components: 5
  umap_min_dist: 0.0
  hdbscan_min_cluster_size: 50
```

| Key | What it does |
|---|---|
| `embedding_model` | Hugging Face model ID for document embeddings. Falls back to `all-MiniLM-L6-v2` if it can't be loaded. |
| `min_topic_size` | Minimum documents per topic. Auto-rescaled when n_docs < 500. |
| `nr_topics` | `"auto"` for automatic reduction, integer for a fixed count, or `null` to disable. Auto-disabled when n_docs < 500. |
| `top_n_words` | Top-N words stored per topic. |
| `umap_*` | UMAP dimensionality reduction parameters. |
| `hdbscan_min_cluster_size` | Minimum HDBSCAN cluster size. Auto-rescaled for small datasets. |

---

## 6.4 `embeddings:`

```yaml
embeddings:
  model_name:       "clinicalbert"   # clinicalbert | biobert | pubmedbert | bert | HF ID
  pooling:          "mean"           # mean (recommended) | cls
  max_length:       512
  stride:           256
  reduce_dims:      true
  n_components:     50
  reduction_method: "pca"            # pca | umap
  device:           null             # null = auto (cuda if available)
```

| Key | What it does |
|---|---|
| `model_name` | Shortcut from `MODEL_REGISTRY` in [src/embeddings.py](../src/embeddings.py), or any HF model ID. |
| `pooling` | Chunk-level pooling strategy. `mean` is the default because CLS tokens in clinical-BERT variants are less informative for long notes. |
| `max_length` | BERT max tokens per chunk. Stay ≤ 512 unless you switch to a long-context model. |
| `stride` | Overlap between chunks when a note exceeds `max_length`. |
| `reduce_dims` | If true, run PCA/UMAP after embedding. |
| `n_components` | Target dimension after reduction. |
| `reduction_method` | `"pca"` (default, deterministic) or `"umap"` (nonlinear). |
| `device` | `"cuda"`, `"cpu"`, or `null` to auto-detect. |

---

## 6.5 `prediction:`

The big one. Controls models, splits, tuning, feature selection, and TF-IDF.

```yaml
prediction:
  target: "readmission_30day"
  test_size: 0.2
  val_size: 0.1
  models:
    - "logistic_regression"
    - "random_forest"
    - "xgboost"
    - "lightgbm"
  feature_types:
    - "tfidf"
    - "topic_distribution"
    - "structured"
    - "embeddings"
    - "combined"
  tuning:
    enabled: true
    n_iter: 30
    cv_folds: 5
    scoring: "roc_auc"
  feature_selection:
    enabled: true
    method: "univariate"
    apply_to: ["tfidf", "combined"]
    k: 200
    score_func: "f_classif"
  tfidf:
    max_features: 5000
    ngram_range: [1, 2]
    min_df: 5
    max_df: 0.95
```

| Key | Default | Notes |
|---|---|---|
| `target` | `readmission_30day` | Label column; currently the only supported task. |
| `test_size` | `0.2` | Final holdout fraction. |
| `val_size` | `0.1` | Validation fraction (cut from the remaining 80%). |
| `models` | 4-model list | Any subset of `logistic_regression`, `random_forest`, `xgboost`, `lightgbm`. |
| `feature_types` | 5 sets | Any subset of `tfidf`, `topic_distribution`, `topic_bertopic`, `structured`, `embeddings`, `text_stats`, `combined`. |

### `prediction.tuning`

| Key | Default | Notes |
|---|---|---|
| `enabled` | `true` | Set `false` for fast smoke runs. |
| `n_iter` | `30` | Random param draws. Each draw is CV'd `cv_folds` times. |
| `cv_folds` | `5` | Stratified k-fold. |
| `scoring` | `"roc_auc"` | Any sklearn scorer string. `"average_precision"` is a good pick for very imbalanced data. |

### `prediction.feature_selection`

| Key | Default | Applies to | Notes |
|---|---|---|---|
| `enabled` | `true` | all methods | Global on/off. |
| `method` | `"univariate"` | all methods | One of `variance`, `univariate`, `l1`, `rfe`, `shap`. |
| `apply_to` | `["tfidf", "combined"]` | all methods | Only these feature sets are reduced. |
| `k` | `200` | `univariate` | Top-k to keep. |
| `score_func` | `"f_classif"` | `univariate` | `"chi2"`, `"f_classif"`, or `"mutual_info"`. |
| `threshold` | — | `variance` | Minimum variance to keep a feature. |
| `C` | — | `l1` | Inverse regularization strength for L1 LogReg. |
| `n_features_to_select` | — | `rfe` | Number of features RFE keeps. |
| `top_k` | — | `shap` | Top-k by mean \|SHAP\| to keep. |

### `prediction.tfidf`

| Key | Default | Notes |
|---|---|---|
| `max_features` | `5000` | Vocabulary cap. |
| `ngram_range` | `[1, 2]` | Unigrams + bigrams. |
| `min_df` | `5` | Drop terms appearing in fewer than this many docs. |
| `max_df` | `0.95` | Drop terms appearing in more than this fraction of docs. |

---

## 6.6 `fairness:`

```yaml
fairness:
  protected_attributes:
    - "gender"
    - "insurance"
    - "age_group"
  metrics:
    - "demographic_parity_difference"
    - "equalized_odds_difference"
    - "false_positive_rate_difference"
    - "false_negative_rate_difference"
```

| Key | What it does |
|---|---|
| `protected_attributes` | Columns in the merged frame to audit. The column names must exist after `merge_dataset`. |
| `metrics` | Fairlearn disparity metrics to compute. Per-group metrics (accuracy, F1, FPR, FNR, selection rate) are always computed. |

To add a new attribute, make sure `data_loader.merge_dataset` produces a column with that name (e.g., derive `ethnicity_bucket`), then add it here — the audit loop picks it up automatically.

---

## 6.7 `visualization:`

```yaml
visualization:
  figure_dpi:    300
  figure_format: "png"
  color_palette: "Set2"
  save_path:     "results/figures/"
```

| Key | What it does |
|---|---|
| `figure_dpi` | Output resolution. 300 is print-quality. |
| `figure_format` | `"png"` (default) or any matplotlib-supported extension. |
| `color_palette` | Seaborn palette name. |
| `save_path` | Where `_save_fig` writes PNGs. The dashboard reads from this exact path. |

---

## 6.8 Environment variables

None. All configuration flows through `config.yaml`. The pipeline is deliberately env-free so notebook runs, CI, and the dashboard all see the same state.
