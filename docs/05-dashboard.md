# 5. Dashboard

The dashboard is a self-contained web app that visualizes the pipeline outputs and optionally runs interactive predictions. It has two halves:

- **Backend** — [dashboard/backend/](../dashboard/backend/) — FastAPI app that serves JSON API routes **and** mounts the frontend as static files.
- **Frontend** — [dashboard/frontend/](../dashboard/frontend/) — plain HTML/CSS/JS pages (no build step) that fetch the API via `js/api.js`.

Everything the UI shows comes from `results/exports/*.json` (written by the exporter) or from a live prediction against the best joblib model. If either is missing, each endpoint falls back to a mock response so the dashboard never crashes.

---

## 5.1 Running it

```bash
# from the project root, with venv activated
uvicorn dashboard.backend.main:app --reload --port 8000
```

Open http://localhost:8000/. The FastAPI app mounts the frontend directory at `/`, so `index.html` is served automatically.

### What needs to be in place for a "real" (non-mock) run

| Needs to exist | Source |
|---|---|
| `results/exports/results.json` | `exporter.export_dashboard_json(prediction_results=...)` |
| `results/exports/fairness.json` | `exporter.export_dashboard_json(fairness_results=...)` |
| `results/exports/topics.json` | `exporter.export_dashboard_json(lda_results=..., readmission_labels=...)` |
| `results/models/*.joblib` | `predict.save_models(...)` (called by the notebook via `run_prediction_pipeline`) |
| `results/figures/*.png` | `visualize.generate_all_figures(...)` |

Run the full notebook once and all of these materialize.

---

## 5.2 Backend layout

```
dashboard/backend/
├── __init__.py
├── main.py          # FastAPI app, routes, static mount
├── inference.py     # ModelRegistry, predict(), explain()
└── schemas.py       # Pydantic request/response models
```

### [main.py](../dashboard/backend/main.py)

- Creates the `FastAPI` app with permissive CORS.
- On startup, calls `inference.registry.load()` to discover the best joblib under `results/models/`.
- Mounts `dashboard/frontend/` as `StaticFiles` (HTML=True so `index.html` is served at `/`).
- Defines the API routes below.

### [inference.py](../dashboard/backend/inference.py)

- `ModelRegistry` — lazy, caches the best model in memory. `discover_best()` looks for `*_BEST.joblib`, falling back to any joblib in the directory. Parses the filename (`{model}__{feature}__BEST.joblib`) into `{model_name, feature_type}`.
- `predict(text, age, gender, insurance, los_days)` — builds a single-row structured feature vector in the exact column order used during training and calls `predict_proba`. For text-based feature types the vectorizer isn't persisted yet, so the current implementation falls back to the structured vector and, failing that, a bounded length-based heuristic. **Treat live text predictions as demonstrative, not clinical.**
- `explain(...)` — builds the same feature vector, calls `compute_shap_values` + `explain_patient`, returns the top-N contributors with their SHAP values and direction.
- `load_results_json / load_fairness_json / load_topics_json` — read-through cache for the exported JSON files, with mock fallbacks.

### [schemas.py](../dashboard/backend/schemas.py)

Pydantic models for request/response bodies. The ones worth knowing:

- `PredictRequest` — `text`, plus optional `age`, `gender`, `insurance`, `los_days`.
- `PredictResponse` — `probability`, `predicted_label`, `risk_level` ∈ {low, moderate, high}, `model_name`, `feature_type`, `threshold`.
- `ExplainResponse` — `probability`, `base_value`, `top_features[]` (`feature`, `value`, `shap`, `direction`), `model_name`.
- `FairnessResponse`, `TopicsResponse`, `ResultsResponse` mirror the shapes produced by `src/exporter.py`.

---

## 5.3 API reference

All endpoints are prefixed with `/api`. Static frontend files are served under `/`.

### `GET /api/health`

```json
{
  "status": "ok",
  "models_loaded": 4,
  "available_results": true
}
```

Use this to confirm the backend started correctly and that the exported JSON + models are visible.

### `GET /api/models`

Lists every joblib in `results/models/`:

```json
{
  "models": [
    {"model": "xgboost", "feature_type": "combined", "is_best": true,
     "path": "results/models/xgboost__combined__BEST.joblib", "size_kb": 1234.5},
    ...
  ]
}
```

### `POST /api/predict`

Request:
```json
{
  "text": "Patient admitted with acute CHF exacerbation ...",
  "age": 72,
  "gender": "M",
  "insurance": "Medicare",
  "los_days": 6.5
}
```

Response:
```json
{
  "probability": 0.6431,
  "predicted_label": 1,
  "risk_level": "high",
  "model_name": "xgboost",
  "feature_type": "structured",
  "threshold": 0.5
}
```

### `POST /api/explain`

Same request body as `/api/predict`. Response:

```json
{
  "probability": 0.6431,
  "base_value": -0.21,
  "top_features": [
    {"feature": "log_los", "value": 1.9, "shap": 0.31, "direction": "increases"},
    {"feature": "age", "value": 72, "shap": 0.14, "direction": "increases"},
    ...
  ],
  "model_name": "xgboost"
}
```

### `GET /api/results`

Serves `results/exports/results.json`. Consumed by the Overview and Compare pages.

### `GET /api/fairness`

Serves `results/exports/fairness.json`. Consumed by the Fairness page.

### `GET /api/topics`

Serves `results/exports/topics.json`. Consumed by the Topics page.

### `GET /api/figures`

Lists PNGs in `results/figures/`:

```json
{
  "figures": [
    {"name": "roc_curves", "filename": "roc_curves.png", "url": "/api/figures/roc_curves.png"},
    ...
  ]
}
```

### `GET /api/figures/{filename}`

Streams a single PNG from `results/figures/` with `media_type="image/png"`. Returns 404 if missing.

---

## 5.4 Frontend layout

```
dashboard/frontend/
├── index.html          # Overview page (entry point)
├── compare.html        # Model comparison
├── fairness.html       # Per-attribute fairness tables + charts
├── topics.html         # LDA topics + readmission rates
├── predict.html        # Live prediction form
├── explain.html        # SHAP explanation view
├── css/
│   ├── reset.css
│   ├── variables.css   # Color + typography tokens
│   ├── layout.css
│   └── components.css
└── js/
    ├── api.js          # fetch wrapper around /api/*
    ├── sidebar.js      # Sidebar navigation
    ├── charts.js       # Chart.js helpers
    └── pages/
        ├── overview.js
        ├── compare.js
        ├── fairness.js
        ├── topics.js
        ├── predict.js
        └── explain.js
```

### Stack notes

- **No build step.** Plain HTML/CSS/JS. Open a page in the browser (served via FastAPI) and it works.
- **Charting.** Chart.js 4.4.1 loaded from jsDelivr in every HTML file.
- **Icons.** Lucide (umd build) via unpkg.
- **Fonts.** Inter + JetBrains Mono via Google Fonts.
- **API base URL.** Defined in `js/api.js`; because the frontend is served by the same FastAPI app, it uses relative `/api/*` paths.

### Page → endpoint mapping

| Page | Endpoints used |
|---|---|
| `index.html` (Overview) | `/api/health`, `/api/results`, `/api/figures` |
| `compare.html` | `/api/results` |
| `fairness.html` | `/api/fairness`, `/api/figures` |
| `topics.html` | `/api/topics`, `/api/figures` |
| `predict.html` | `/api/predict` |
| `explain.html` | `/api/explain` |

---

## 5.5 Extending the dashboard

- **Add a metric to the Overview page.** Add it to `build_results_payload` in [src/exporter.py](../src/exporter.py), re-run the notebook (or just `exporter.export_dashboard_json(...)`), then render it in `js/pages/overview.js`.
- **Add a new protected attribute.** Add it to `config.fairness.protected_attributes`. `run_fairness_audit` picks it up automatically, `build_fairness_payload` serializes it, and the Fairness page renders whatever attributes it receives.
- **Persist TF-IDF vectorizer for live text predictions.** Currently `dashboard/backend/inference.py` only builds structured features for live prediction. Serialize the fitted vectorizer alongside the model in `predict.save_models`, load it in `ModelRegistry`, and branch on `feat_type == "tfidf"` to transform the incoming text before `predict_proba`.
- **Add a new page.** Create `dashboard/frontend/newpage.html`, a `js/pages/newpage.js`, and a FastAPI route if you need a new data source.

---

## 5.6 Security / deployment caveats

- CORS is wildcarded (`allow_origins=["*"]`) for local development. Narrow this before exposing the app beyond localhost.
- Predictions are demonstrative on synthetic data. The dashboard is **not** a medical device and **must not** drive real clinical decisions without separate validation.
- `results/exports/` and `results/models/` contain only aggregate metrics and trained weights — no patient text — but treat them as sensitive anyway if they came from real MIMIC-IV data.
