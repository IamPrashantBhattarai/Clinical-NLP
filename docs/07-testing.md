# 7. Testing & Troubleshooting

The project ships a single script, [tests/test_pipeline.py](../tests/test_pipeline.py), that runs every module end-to-end as a smoke test. It's deliberately plain — no `pytest` dependency — so it's cheap to run anywhere.

---

## 7.1 Running the tests

From the project root, with the venv activated:

```bash
python tests/test_pipeline.py
```

Each test prints `[PASS]` or `[FAIL]` with a short `>>` summary. A failing test prints a full traceback. The script exits with status `1` if any test failed.

### Faster subsets

Tests are grouped by `SECTION` headers inside `main()`. The quickest way to run a single section is to comment out the others — the script is intentionally simple, and import time is the main cost.

### GPU / heavy models

- The embedding tests (`test_load_embedding_model`, `test_embed_*`) download Bio_ClinicalBERT weights (~440 MB) on first run and cache them under `~/.cache/huggingface/`. Subsequent runs are fast.
- BERTopic is only exercised indirectly via feature engineering; `test_pipeline.py` doesn't retrain BERTopic inside the test run.

---

## 7.2 What's covered

| Section | Tests | Exercises |
|---|---|---|
| **1** — Synthetic data | `test_generate_patients`, `test_generate_admissions`, `test_generate_notes`, `test_run_generates_csv_files` | [src/generate_synthetic_data.py](../src/generate_synthetic_data.py) |
| **2** — Data loader | `test_load_discharge_notes`, `test_load_admissions`, `test_load_patients`, `test_create_readmission_label`, `test_merge_dataset`, `test_load_all_convenience`, `test_get_data_summary` | [src/data_loader.py](../src/data_loader.py) |
| **3** — Preprocessing | `test_clean_clinical_text`, `test_extract_sections`, `test_remove_sections`, `test_tokenize_clinical`, `test_create_bigrams_trigrams`, `test_full_pipeline`, `test_bow_corpus`, `test_tfidf_matrix` | [src/preprocess.py](../src/preprocess.py) + TF-IDF feature builder |
| **4** — Prediction | `test_split_data`, `test_get_model`, `test_train_and_evaluate`, `test_cross_validate`, `test_feature_importance`, `test_optimal_threshold`, `test_tune_hyperparameters`, `test_full_prediction_pipeline`, `test_prediction_pipeline_with_tuning` | [src/predict.py](../src/predict.py) |
| **4.5** — SHAP | `test_shap_global_importance`, `test_shap_patient_explanation`, `test_run_shap_analysis` | [src/explainability.py](../src/explainability.py) |
| **4.6** — Feature selection | `test_variance_threshold_selection`, `test_univariate_selection`, `test_l1_selection`, `test_rfe_selection`, `test_shap_selection`, `test_select_features_dispatcher`, `test_pipeline_with_feature_selection` | [src/feature_selection.py](../src/feature_selection.py) + predict pipeline wiring |
| **5** — Fairness | `test_group_metrics`, `test_fairness_metrics`, `test_fairness_audit` | [src/fairness.py](../src/fairness.py) |
| **6** — Embeddings | `test_load_embedding_model`, `test_embed_single_texts`, `test_embed_long_text_chunking`, `test_reduce_embeddings` | [src/embeddings.py](../src/embeddings.py) |
| **7** — Visualization | `test_plot_demographics`, `test_plot_note_length`, `test_plot_roc_pr_confusion`, `test_plot_fairness` | [src/visualize.py](../src/visualize.py) |

The `_build_small_feature_sets` helper inside `test_pipeline.py` assembles a miniature `feature_sets` dict so the prediction, SHAP, and feature-selection tests run in seconds without needing a full LDA/BERTopic fit.

### What isn't covered yet

- **BERTopic training** — heavy to run as a unit test; covered indirectly via the notebook and by the 1-D probs fallback in `create_topic_features`.
- **Dashboard endpoints** — exercised by hand (`curl http://localhost:8000/api/health`). An httpx-based FastAPI test client would slot in cleanly under a new `tests/test_dashboard.py` if you need it.
- **Exporter round-trip** — no dedicated test yet; the notebook's dashboard export cell is the current smoke test.

---

## 7.3 Troubleshooting cheatsheet

### Import errors

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root, not from inside `tests/`. |
| `ModuleNotFoundError: No module named 'scispacy'` | scispaCy is optional. Call preprocessing with `use_scispacy=False` or install it per [docs/02-setup.md](02-setup.md). |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` |
| `LookupError: Resource ... not found` (NLTK) | First call to `_ensure_nltk()` downloads these lazily. If you're offline, run `python -c "import nltk; [nltk.download(r) for r in ['stopwords','wordnet','punkt','punkt_tab','averaged_perceptron_tagger']]"`. |

### Runtime errors

| Symptom | Cause + fix |
|---|---|
| `IndexError: tuple index out of range` in `create_topic_features` | Old `feature_engineer.py` cached in kernel. Restart the Jupyter kernel; the current code handles 1-D BERTopic `probs`. |
| `ValueError: Found array with 0 sample(s)` | The `readmission_30day >= 0` filter dropped everything. Check `create_readmission_label` — most synthetic runs should yield ~70-80% eligible. |
| BERTopic raises `ValueError` on small datasets | `train_bertopic` already rescales `min_topic_size` and disables `nr_topics='auto'` for n<500. If it still fails, pass `bertopic_model=None` to `build_feature_sets`. |
| `MemoryError` / OOM during XGBoost or RF training on Windows | Models use `n_jobs=1` by default for this reason — don't override it. See commit `62922ed`. |
| `FileNotFoundError: data/synthetic_discharge.csv` | Run `generate_data()` (notebook Section 1) before loading. |
| LightGBM warnings about `is_unbalance` + `scale_pos_weight` | Harmless; LightGBM picks one. |

### Dashboard issues

| Symptom | Fix |
|---|---|
| Dashboard starts but every page shows mock data | `results/exports/*.json` missing. Re-run the notebook through Section 8 (`exporter.export_dashboard_json(...)`). |
| `/api/predict` always returns a demo prediction | No joblib found under `results/models/`. Re-run the prediction section — `run_prediction_pipeline` writes models via `save_models`. |
| `/api/explain` returns the mock list | SHAP import failed (look at backend logs) or the best model's feature schema doesn't match the 12-column structured vector the live endpoint builds. |
| Frontend loads but charts are empty | Check the browser console — likely a 404 on `/api/figures/*.png` because the figures directory is empty. Re-run `generate_all_figures`. |
| CORS blocked | Serving the frontend through `uvicorn` (not `file://`) sidesteps this. The backend also sets `allow_origins=["*"]`. |

### Notebook kernel hygiene

If you edit a module and the notebook doesn't pick it up:

```python
import importlib, src.feature_engineer
importlib.reload(src.feature_engineer)
from src.feature_engineer import build_feature_sets
```

Or restart the kernel entirely (Kernel → Restart). Stale imports caused the BERTopic `IndexError` you saw when we first wired in feature selection.

---

## 7.4 Continuous integration notes

No CI is configured in this repo yet. If you add GitHub Actions:

- Run `pip install -r requirements.txt` + `python -m spacy download en_core_web_sm` + NLTK downloads in a setup step.
- Skip the embedding tests on CI or cache `~/.cache/huggingface/` — the ClinicalBERT download is ~440 MB and will dominate the build time.
- Use `pytest tests/test_pipeline.py` with a small shim, or just invoke `python tests/test_pipeline.py` and fail the job on non-zero exit.
- The visualization tests write to `results/figures/` — either clean that up in a post-step or mark the directory as an artifact.
