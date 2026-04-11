# Clinical NLP — 30-Day Readmission Prediction

End-to-end NLP pipeline that predicts 30-day hospital readmission from discharge summaries, with topic modeling, fairness auditing, SHAP explainability, and a FastAPI dashboard for exploring results.

## Highlights

- **Data** — Works with real MIMIC-IV discharge notes or the bundled synthetic MIMIC-style generator (`src/generate_synthetic_data.py`).
- **Preprocessing** — PHI scrubbing, NLTK tokenization, clinical stopwords, optional scispaCy.
- **Clinical embeddings** — Bio_ClinicalBERT / BioBERT / PubMedBERT with sliding-window chunking for long notes and optional PCA compression.
- **Topic modeling** — LDA (Gensim, coherence-based topic-count selection) and BERTopic (BioBERT + UMAP + HDBSCAN) with a per-topic readmission association test.
- **Feature engineering** — TF-IDF, LDA/BERTopic distributions, dense clinical embeddings, structured demographics/utilization, text statistics, and a combined stack.
- **Feature selection** — Variance threshold, univariate, L1, RFE, and SHAP-based selectors applied to high-dimensional feature sets (TF-IDF / combined).
- **Modeling** — Logistic Regression, Random Forest, XGBoost, LightGBM trained on each feature set with RandomizedSearchCV tuning and SMOTE/class-weight support.
- **Explainability** — SHAP global importance plus per-patient local explanations ([src/explainability.py](src/explainability.py)).
- **Fairness** — Fairlearn audit over gender, insurance, and age group (demographic parity, equalized odds, FPR/FNR, selection rate).
- **Dashboard** — FastAPI backend at [dashboard/backend/](dashboard/backend/) serving a static frontend under [dashboard/frontend/](dashboard/frontend/) with overview, topics, fairness, explain, compare, and predict pages.

## Quickstart

```bash
# 1. Environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run the pipeline end-to-end
jupyter notebook notebooks/full_pipeline.ipynb

# 3. Serve the dashboard (after the notebook has written results/exports/*.json)
uvicorn dashboard.backend.main:app --reload --port 8000
# Open http://localhost:8000/
```

The notebook generates 200 synthetic patients by default — no MIMIC-IV credentials required.

## Documentation

Full docs live under [docs/](docs/):

| Doc | What it covers |
|---|---|
| [docs/01-overview.md](docs/01-overview.md) | Goals, high-level architecture, pipeline diagram |
| [docs/02-setup.md](docs/02-setup.md) | Install, data options, running the pipeline |
| [docs/03-pipeline.md](docs/03-pipeline.md) | Module-by-module reference for `src/` |
| [docs/04-models-and-evaluation.md](docs/04-models-and-evaluation.md) | Models, tuning, feature selection, SHAP, fairness |
| [docs/05-dashboard.md](docs/05-dashboard.md) | FastAPI backend, frontend, API reference |
| [docs/06-configuration.md](docs/06-configuration.md) | Full `config/config.yaml` reference |
| [docs/07-testing.md](docs/07-testing.md) | Running tests, common failure modes |

## Repository layout

```
ClinicalNLP/
├── config/config.yaml                  # Central pipeline configuration
├── data/                               # Raw + synthetic MIMIC-style tables
├── notebooks/full_pipeline.ipynb       # End-to-end demo notebook
├── src/
│   ├── generate_synthetic_data.py      # Synthetic MIMIC-IV generator
│   ├── data_loader.py                  # Table loading + merging
│   ├── preprocess.py                   # PHI scrub, tokenization, stopwords
│   ├── embeddings.py                   # ClinicalBERT / BioBERT embeddings
│   ├── topic_model.py                  # LDA + BERTopic
│   ├── feature_engineer.py             # TF-IDF / topics / structured / stats / combined
│   ├── feature_selection.py            # Variance / univariate / L1 / RFE / SHAP
│   ├── predict.py                      # Train + tune + evaluate all models
│   ├── explainability.py               # SHAP global + per-patient
│   ├── fairness.py                     # Fairlearn audit
│   ├── visualize.py                    # Publication figures → results/figures/
│   └── exporter.py                     # Dashboard JSON exporter
├── dashboard/
│   ├── backend/                        # FastAPI app (main, inference, schemas)
│   └── frontend/                       # Static HTML/CSS/JS pages
├── results/
│   ├── figures/                        # PNG outputs
│   ├── models/                         # Pickled best models
│   └── exports/                        # Dashboard JSON (results, fairness, topics)
└── tests/test_pipeline.py              # Unit + integration tests
```

## Tests

```bash
python tests/test_pipeline.py
```

Covers preprocessing, embeddings, topic modeling, feature engineering, feature selection, prediction, SHAP, and fairness. See [docs/07-testing.md](docs/07-testing.md).

## License & data notice

This repository contains no real patient data. Synthetic data is generated locally via `generate_synthetic_data.py` and is **not** a substitute for a MIMIC-IV data-use agreement if you intend to run on real records.
