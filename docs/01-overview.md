# 1. Project Overview

## Goal

Predict whether a hospital patient will be readmitted within 30 days of discharge, using free-text discharge summaries plus structured admission data, and do so with:

- **Reproducible end-to-end automation** — synthetic data in, figures + metrics + dashboard JSON out.
- **Clinically-aware NLP** — clinical BERT variants, medical stopwords, section-aware preprocessing, topic labeling against a clinical keyword vocabulary.
- **Interpretability** — SHAP-based global and per-patient explanations so predictions can be scrutinized.
- **Fairness awareness** — explicit group-level auditing across gender, insurance, and age group using Fairlearn.

## High-level architecture

```
            ┌─────────────────────────┐
            │  Data generation / load │  src/generate_synthetic_data.py
            │  (MIMIC-IV or synthetic)│  src/data_loader.py
            └────────────┬────────────┘
                         │  merged DataFrame
                         ▼
            ┌─────────────────────────┐
            │     Preprocessing       │  src/preprocess.py
            │  PHI scrub + tokens     │
            └────────────┬────────────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
┌───────────┐  ┌──────────────────┐  ┌────────────────┐
│ Clinical  │  │  Topic modeling  │  │  Feature eng.  │
│ embeddings│  │  (LDA + BERTopic)│  │  (TF-IDF, ...) │
│ src/embed │  │  src/topic_model │  │  src/feature_* │
└─────┬─────┘  └────────┬─────────┘  └───────┬────────┘
      │                 │                    │
      └─────────────────┴────────────────────┘
                         │  feature_sets dict
                         ▼
            ┌─────────────────────────┐
            │  Prediction pipeline    │  src/predict.py
            │  (4 models × feats)     │  + feature_selection
            │  RandomizedSearchCV     │
            └────────────┬────────────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
┌───────────┐  ┌──────────────────┐  ┌────────────────┐
│  SHAP     │  │  Fairness audit  │  │  Visualization │
│ explain.  │  │  (Fairlearn)     │  │  figures       │
└─────┬─────┘  └────────┬─────────┘  └───────┬────────┘
      └─────────────────┴────────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │    exporter.py          │  → results/exports/*.json
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │   FastAPI dashboard     │  dashboard/backend + frontend
            └─────────────────────────┘
```

## Data flow (one run)

1. **Load** merged cohort (`df`) — discharge notes + admissions + patients, with derived fields (`readmission_30day`, `los_days`, `age`).
2. **Preprocess** into `processed` — adds `cleaned_text`, `tokens`, `num_tokens`, etc.
3. **Embed** `processed["cleaned_text"]` → `embedding_matrix` (n_docs × d).
4. **Topic model** on eligible tokens → `lda_results` and `bertopic_model`.
5. **Feature engineer** → `feature_sets` (TF-IDF / topic_lda / topic_bertopic / embeddings / structured / text_stats / combined / label).
6. **Predict** → `prediction_results` with per-(model, feature) metrics, best model, tuned params, feature-selection info, trained models.
7. **Explain** best model → `shap_results`.
8. **Audit fairness** over protected attrs → `fairness_results`.
9. **Visualize** → PNGs under `results/figures/`.
10. **Export** JSON → `results/exports/{results,fairness,topics}.json` for the dashboard.

## Design principles

- **Config-first.** Almost every knob (model list, topic counts, tuning iterations, feature-selection method, fairness attrs) lives in `config/config.yaml`. See [docs/06-configuration.md](06-configuration.md).
- **Graceful degradation.** BERTopic, scispaCy, GPU, and even trained models are optional — the pipeline and dashboard fall back to reasonable defaults or mock data when anything is missing.
- **Small-dataset safety.** BERTopic auto-scales `min_topic_size`, HDBSCAN parameters, and disables `nr_topics='auto'` when there are <500 docs. Preprocessing guards against empty notes; feature_engineer realigns shapes when eligible subsets differ from the full frame.
- **Sparse-aware.** TF-IDF and combined matrices stay sparse through feature selection and splitting (`scipy.sparse.hstack`, `csr_matrix`) so memory stays bounded on the full vocabulary.

## What's in scope vs. out

| In scope | Out of scope |
|---|---|
| Readmission prediction from free text + light structured data | Time-to-event / survival modeling |
| Single-task binary classification (30-day readmit) | Multi-label outcome prediction |
| Fairness auditing (post-hoc) | In-training mitigation (reweighing, adversarial debiasing) |
| SHAP for tree + linear models | Counterfactual explanations, LIME |
| Static dashboard backed by exported JSON | Live model retraining from the UI |

See [docs/03-pipeline.md](03-pipeline.md) for a module-by-module walkthrough and [docs/04-models-and-evaluation.md](04-models-and-evaluation.md) for the modeling, tuning, and evaluation details.
