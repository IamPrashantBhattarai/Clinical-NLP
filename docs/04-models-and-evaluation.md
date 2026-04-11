# 4. Models & Evaluation

This page covers the modeling side of the pipeline: which models are trained, how they're tuned, how feature selection is applied, how explanations are computed, and how fairness is audited.

---

## 4.1 Models

[src/predict.py](../src/predict.py) trains four classifiers (configurable via `prediction.models`):

| Model | Class | Defaults |
|---|---|---|
| `logistic_regression` | `sklearn.linear_model.LogisticRegression` | `solver="saga"`, `penalty="l2"`, `C=1.0`, `class_weight="balanced"`, `max_iter=1000` |
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=200`, `min_samples_leaf=5`, `class_weight="balanced"` |
| `xgboost` | `xgboost.XGBClassifier` | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric="logloss"` |
| `lightgbm` | `lightgbm.LGBMClassifier` | `n_estimators=200`, `num_leaves=31`, `is_unbalance=True` |

All estimators use `n_jobs=1` on Windows to avoid an OOM / fork issue that surfaced during development — see commit `62922ed`. This is safe because `RandomizedSearchCV` already parallelizes across folds.

### Class imbalance

Three knobs, any combination of which can be active:

1. **`class_weight="balanced"`** on logistic regression and RF
2. **`is_unbalance=True`** on LightGBM
3. **SMOTE** via `apply_smote` in [src/predict.py:89](../src/predict.py#L89) — applied only to the training split inside `train_model` when enabled

The pipeline defaults rely on class weights / `is_unbalance`, which is cheaper than SMOTE on sparse TF-IDF matrices.

---

## 4.2 Feature sets that get trained

Controlled by `prediction.feature_types`:

| Key | Where built | What it is |
|---|---|---|
| `tfidf` | `create_tfidf_features` | Sparse n-gram bag-of-words, `max_features=5000`, `ngram_range=(1,2)`, `sublinear_tf=True` |
| `topic_distribution` | `create_topic_features(..., "lda")` | Dense LDA doc-topic probabilities |
| `topic_bertopic` | `create_topic_features(..., "bertopic")` | Dense BERTopic distribution (or one-hot fallback) |
| `embeddings` | `embeddings.embed_texts` + optional PCA | Dense clinical-BERT vectors |
| `structured` | `create_structured_features` | Dense demographics + utilization |
| `text_stats` | `create_text_statistics_features` | Dense note-level statistics |
| `combined` | `combine_features` | Sparse hstack of TF-IDF + structured + text_stats |

For every `(feature_type, model)` pair, the pipeline computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC on the held-out test set and picks the best by the configured scoring metric (default ROC-AUC).

---

## 4.3 Hyperparameter tuning

Controlled by `prediction.tuning`:

```yaml
tuning:
  enabled: true
  n_iter: 30           # random draws from PARAM_GRIDS
  cv_folds: 5
  scoring: "roc_auc"
```

`RandomizedSearchCV` is run inside [tune_hyperparameters](../src/predict.py#L219) with a stratified k-fold. The chosen grids live in `PARAM_GRIDS` at [src/predict.py:185](../src/predict.py#L185); each grid spans the typical useful range per model (e.g., `C` across four orders of magnitude for LogReg, `max_depth` 3–10 for XGBoost).

The best estimator is refit on the full train split, then evaluated on val/test. The winning params + CV score land in `prediction_results["tuned_params"]` keyed by `(model, feature_type)`.

---

## 4.4 Feature selection

Controlled by `prediction.feature_selection`:

```yaml
feature_selection:
  enabled: true
  method: "univariate"          # variance | univariate | l1 | rfe | shap
  apply_to: ["tfidf", "combined"]
  k: 200                        # univariate only
  score_func: "f_classif"       # univariate only
  # threshold: 0.0              # variance
  # C: 1.0                      # l1
  # n_features_to_select: 50    # rfe
  # top_k: 50                   # shap
```

**How it wires in.** Inside [run_prediction_pipeline](../src/predict.py#L607), for every feature type listed in `apply_to`:

1. Split `(X, y)` into train/val/test (stratified).
2. Call `select_features` on **train only** — never on val/test — to fit the selector.
3. Apply the resulting `indices` to val and test via `apply_selection` so the columns stay aligned.
4. Replace `feat_names` with the selected names.
5. Record `{method, n_before, n_after, indices, names}` under `prediction_results["feature_selection"][feature_type]`.

Failures are caught and logged — if, say, L1 eliminates everything on a pathological split, the pipeline falls back to the unreduced features and keeps going.

Why limit it to TF-IDF and combined? Those are the only high-dimensional sets; topic distributions and structured features already have <30 columns.

### Methods cheat-sheet

| Method | When it helps | Caveats |
|---|---|---|
| `variance` | Quickly prune constant / near-constant cols from TF-IDF on small vocab | No supervision signal |
| `univariate` (`f_classif`) | Default. Fast, label-aware, works for sparse TF-IDF | Assumes features are roughly independent of each other |
| `univariate` (`chi2`) | Alt for non-negative features | Requires non-negative matrix (fine for TF-IDF) |
| `l1` | Sparse logistic regression as selector | Unstable on very wide matrices with tiny samples |
| `rfe` | Exhaustive ranking via repeated model fitting | Slow — O(n_features × k) |
| `shap` | Use a trained model's attributions as selector | Expensive; trains a model just for selection |

---

## 4.5 Evaluation metrics

Every row of `prediction_results["results_df"]` has:

- **accuracy / precision / recall / F1** — at a 0.5 threshold (or an optimized threshold from `find_optimal_threshold` if you use it downstream)
- **ROC-AUC** — threshold-free ranking quality, robust to class imbalance
- **PR-AUC (average precision)** — more informative than ROC-AUC when the positive class is rare

The best model is selected by `prediction.tuning.scoring` (default ROC-AUC). Inspect `prediction_results["cv_results"]` to see the CV performance of the winning model vs. the test performance — a big gap flags overfitting, which is easy to hit on ~200 synthetic notes.

---

## 4.6 SHAP explainability

[src/explainability.py](../src/explainability.py) picks an explainer automatically:

- **Tree models** (RF, XGBoost, LightGBM) → `shap.TreeExplainer` — fast, exact
- **Logistic Regression** → `shap.LinearExplainer`

### What `run_shap_analysis` returns

```python
{
  "model_name":        "xgboost",
  "feature_type":      "combined",
  "base_value":        float,
  "global_importance": pd.DataFrame,   # feature, mean_abs_shap
  "shap_values":       np.ndarray,     # (n_explain, n_features)
  "X_explain":         np.ndarray,     # samples used for the explanation
  "feature_names":     [str, ...],
  "patient_examples":  [                # top-N risk patients
      {"patient_idx", "base_value", "predicted_logit", "top_contributors": [...]},
      ...
  ],
}
```

The notebook prints the global top-20 features (surface in `results/figures/shap_global_importance.png`) and three per-patient breakdowns (`shap_patient_*.png`). These figures also flow into the dashboard's `/explain` page.

### Reading SHAP outputs clinically

- A positive SHAP value **pushes** the prediction toward "readmitted"; negative pulls away.
- `base_value` is the model's average log-odds — the starting point before any feature contributes.
- For TF-IDF features, the "feature" is an n-gram. If `sepsis` has a high positive SHAP, the model learned that mentioning sepsis increases 30-day risk.
- Watch out for data-leakage n-grams like `follow up in` or `discharge` — this is why `preprocessing.remove_sections` strips `Discharge *` sections.

---

## 4.7 Fairness audit

[src/fairness.py](../src/fairness.py) uses Fairlearn and loops over every attribute in `config.fairness.protected_attributes` (default: `gender`, `insurance`, `age_group`).

### Per-group metrics (via `MetricFrame`)

For each group inside each attribute:

| Metric | Meaning |
|---|---|
| accuracy | Overall correctness in the group |
| precision | Of predicted readmits in the group, how many were actual readmits |
| recall | Of actual readmits in the group, how many were caught |
| f1 | Harmonic mean of precision and recall |
| fpr | False positive rate |
| fnr | False negative rate (from `1 - recall`) |
| selection_rate | Fraction predicted positive (used for demographic parity) |

### Disparity metrics

For the attribute as a whole:

- **demographic_parity_difference** — max − min of selection rate across groups
- **equalized_odds_difference** — max group-level disparity in TPR or FPR
- **fpr_difference / fnr_difference** — explicit gaps

### What to look at

- **Large `fnr_difference` on `insurance`** → the model misses readmissions more often for certain insurance types. High-stakes failure mode.
- **Large `demographic_parity_difference` on `gender`** → raw flag rates differ between genders. Not automatically bad (base rates may differ), but worth pairing with equalized odds.
- **Small sample sizes on synthetic data** make many of these metrics noisy — take the synthetic-run numbers as a smoke test, not a verdict.

The audit is post-hoc: we currently don't apply in-training mitigation. If you need that, wrap models with Fairlearn's `ExponentiatedGradient` reduction or add a `ThresholdOptimizer` on the held-out set.

---

## 4.8 Where to look next

- **Notebook** — [notebooks/full_pipeline.ipynb](../notebooks/full_pipeline.ipynb) sections 5, 6, 6.5 walk through the prediction results, fairness audit, and SHAP explanations interactively.
- **Tests** — [tests/test_pipeline.py](../tests/test_pipeline.py) sections 4, 4.5, 4.6, 5 cover the prediction pipeline, SHAP, feature selection, and fairness.
- **Config reference** — [docs/06-configuration.md](06-configuration.md) has every knob and its default.
