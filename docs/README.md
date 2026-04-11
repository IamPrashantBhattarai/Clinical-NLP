# Clinical NLP — Documentation

Full docs for the Clinical NLP readmission prediction pipeline. Start with the [project README](../README.md) for a quickstart, then dive into the sections below.

## Contents

1. [Overview](01-overview.md) — goals, architecture diagram, data flow, design principles.
2. [Setup & Running](02-setup.md) — install, data options, running the notebook or programmatically, output layout.
3. [Pipeline Reference](03-pipeline.md) — module-by-module walkthrough of everything in [src/](../src/).
4. [Models & Evaluation](04-models-and-evaluation.md) — models, tuning, feature selection, SHAP, fairness audit.
5. [Dashboard](05-dashboard.md) — FastAPI backend, frontend pages, full API reference.
6. [Configuration Reference](06-configuration.md) — every key in `config/config.yaml`.
7. [Testing & Troubleshooting](07-testing.md) — test suite layout and common failure modes.

## Reading order

- **New to the repo?** README → 01 → 02 → run the notebook → skim 03.
- **Extending the pipeline?** 03 (pipeline reference) → 04 (models/eval) → 06 (config).
- **Wiring up the dashboard?** 05 (dashboard) → 02 (setup) → 07 (troubleshooting).
- **Debugging a specific failure?** 07 (troubleshooting cheatsheet) → the relevant module in 03.
