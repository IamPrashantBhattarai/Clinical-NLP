"""
FastAPI app for the Clinical NLP dashboard.

Run from project root:
    uvicorn dashboard.backend.main:app --reload --port 8000

Then open http://localhost:8000/
"""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Make project root importable so `src.*` works
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.backend import inference, schemas

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = ROOT / "dashboard" / "frontend"
FIGURES_DIR = ROOT / "results" / "figures"

app = FastAPI(
    title="Clinical NLP Dashboard API",
    description="Readmission prediction, SHAP explanations, fairness audit, and topic modeling.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    logger.info("Loading model registry ...")
    inference.registry.load()
    meta = inference.registry.best_meta
    if meta:
        logger.info("Best model: %s + %s", meta.get("model_name"), meta.get("feature_type"))
    else:
        logger.warning("No models loaded — dashboard will run in demo mode.")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=schemas.HealthResponse)
def health():
    models = inference.registry.list_all()
    return {
        "status": "ok",
        "models_loaded": len(models),
        "available_results": (ROOT / "results" / "exports" / "results.json").exists(),
    }


@app.get("/api/models")
def list_models():
    return {"models": inference.registry.list_all()}


@app.post("/api/predict", response_model=schemas.PredictResponse)
def predict(req: schemas.PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    out = inference.predict(
        text=req.text,
        age=req.age,
        gender=req.gender,
        insurance=req.insurance,
        los_days=req.los_days,
    )
    return out


@app.post("/api/explain", response_model=schemas.ExplainResponse)
def explain(req: schemas.PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    out = inference.explain(
        text=req.text,
        age=req.age,
        gender=req.gender,
        insurance=req.insurance,
        los_days=req.los_days,
        top_n=10,
    )
    return out


@app.get("/api/results")
def get_results():
    return inference.load_results_json()


@app.get("/api/fairness")
def get_fairness():
    return inference.load_fairness_json()


@app.get("/api/topics")
def get_topics():
    return inference.load_topics_json()


@app.get("/api/figures")
def list_figures():
    if not FIGURES_DIR.exists():
        return {"figures": []}
    figures = [
        {"name": p.stem, "filename": p.name, "url": f"/api/figures/{p.name}"}
        for p in sorted(FIGURES_DIR.glob("*.png"))
    ]
    return {"figures": figures}


@app.get("/api/figures/{filename}")
def get_figure(filename: str):
    path = FIGURES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Figure not found: {filename}")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Static frontend (mount last so /api routes take precedence)
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return JSONResponse({"detail": f"Frontend directory missing: {FRONTEND_DIR}"})
