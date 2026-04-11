# 2. Setup & Running

## 2.1 Prerequisites

- **Python** 3.10 or 3.11 (3.12 works but some ML libs lag on new releases)
- **OS** Windows 10/11, macOS, or Linux. Development is done on Windows 11; paths in docs use forward slashes.
- **Memory** ~4 GB free (synthetic data), 16 GB+ if running on a full MIMIC-IV subset.
- **GPU** Optional. ClinicalBERT embeddings auto-detect CUDA — CPU works, just slower.

## 2.2 Install

```bash
git clone https://github.com/IamPrashantBhattarai/Clinical-NLP.git ClinicalNLP
cd ClinicalNLP

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# spaCy model (the pipeline uses en_core_web_sm by default)
python -m spacy download en_core_web_sm

# Optional: scispaCy (enables use_scispacy=True in preprocess)
# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

NLTK corpora (`stopwords`, `wordnet`, `punkt`, `punkt_tab`, `averaged_perceptron_tagger`) are downloaded lazily by [src/preprocess.py](../src/preprocess.py) on first use.

## 2.3 Data options

### A. Synthetic (default — no credentials required)

```python
from src.generate_synthetic_data import run as generate_data
generate_data(output_dir="data", n_patients=200)
```

This writes:
- `data/synthetic_discharge.csv`
- `data/synthetic_admissions.csv`
- `data/synthetic_patients.csv`

Load it with `load_all(config, use_synthetic=True)` (the notebook does this).

### B. Real MIMIC-IV

Place the CSVs that match the paths in `config/config.yaml` under `data:`:
- `data/discharge.csv`
- `data/admissions.csv`
- `data/patients.csv`

Then load with `load_all(config, use_synthetic=False)`. A MIMIC-IV credentialed access is required — the repository ships none.

## 2.4 Running the pipeline

### Option 1 — Notebook (recommended)

```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

The notebook walks through every stage (data → preprocess → embeddings → LDA → BERTopic → features → prediction → SHAP → fairness → figures → dashboard export). Run cells top to bottom on a fresh kernel.

### Option 2 — Programmatic

```python
import yaml
from src.data_loader import load_all
from src.preprocess import build_preprocessing_pipeline
from src.embeddings import embed_texts
from src.topic_model import run_lda_pipeline, train_bertopic
from src.feature_engineer import build_feature_sets
from src.predict import run_prediction_pipeline
from src.fairness import run_fairness_audit
from src.explainability import run_shap_analysis
from src.visualize import generate_all_figures
from src.exporter import export_dashboard_json

config = yaml.safe_load(open("config/config.yaml"))

df         = load_all(config, use_synthetic=True)
processed  = build_preprocessing_pipeline(df, config=config)
emb_matrix, _, _ = embed_texts(processed["cleaned_text"].tolist(), config=config)

eligible   = processed[processed["readmission_30day"] >= 0].reset_index(drop=True)
tokens     = eligible["tokens"].tolist()
lda_res    = run_lda_pipeline(tokens, config=config)
bt_model, _, _ = train_bertopic(eligible["cleaned_text"].tolist(), config=config)

feats      = build_feature_sets(processed, lda_results=lda_res,
                                bertopic_model=bt_model, config=config,
                                embedding_matrix=emb_matrix)
pred       = run_prediction_pipeline(feats, config=config)
fairness   = run_fairness_audit(pred, processed, config=config)
shap_res   = run_shap_analysis(pred, feats, config=config)

generate_all_figures(merged_df=df, prediction_results=pred, shap_results=shap_res,
                     lda_results=lda_res, fairness_results=fairness, config=config)

export_dashboard_json(prediction_results=pred, fairness_results=fairness,
                      lda_results=lda_res,
                      readmission_labels=eligible["readmission_30day"].values)
```

## 2.5 Output layout

After a full run:

```
results/
├── figures/                # publication-quality PNGs
│   ├── demographics.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── topic_word_clouds.png
│   ├── fairness_group_metrics.png
│   ├── fairness_disparity.png
│   ├── shap_summary.png
│   ├── shap_global_importance.png
│   └── shap_patient_*.png
├── models/                 # best + all trained joblib files
└── exports/                # dashboard JSON
    ├── results.json
    ├── fairness.json
    └── topics.json
```

## 2.6 Common gotchas

| Symptom | Fix |
|---|---|
| `IndexError: tuple index out of range` in `create_topic_features` | You have an old `feature_engineer.py` cached in the kernel; restart the Jupyter kernel. The current code handles 1-D `probs` from BERTopic. |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm`. |
| BERTopic training raises HDBSCAN errors on ~200 docs | Already handled: `train_bertopic` rescales parameters for n<500. If it still fails, set `bertopic_model=None` in `build_feature_sets` — the rest of the pipeline skips BERTopic features cleanly. |
| `No trained models found` on dashboard startup | Re-run the notebook through the prediction section so `results/models/*.joblib` files exist, or accept the mock fallback. |
| Notebook is slow on first embedding call | Hugging Face downloads ClinicalBERT weights (~440 MB) on first use and caches under `~/.cache/huggingface/`. |

See [docs/07-testing.md](07-testing.md) for troubleshooting individual modules via the test suite.
