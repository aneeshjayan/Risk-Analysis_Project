# Credit Risk Decision Intelligence

**FSE 570 — Data Science Capstone | West Virginia Team**

Predicts Probability of Default (PD) on LendingClub loans using ensemble ML models,
explains decisions with SHAP, and surfaces insights via an LLM chatbot and interactive dashboard.

## Team

| Member | Role | Module |
|---|---|---|
| Subramanian Raj Narayanan | Project Coordinator / Technical Lead | SHAP + LLM Integration |
| Aneesh Jayan Prabhu | Data & Modelling Lead | XGBoost |
| Balamadhan Sivaraman | Engineering Lead | LightGBM |
| Kalaivani Ravichandran | Documentation Lead | Dashboard + Random Forest |
| Jayasurya Sakthivel | Data & Modelling Lead | Preprocessing + Evaluation |

## Architecture

```
Lending_club_data.csv
        │
        ▼
src/data/ingestion.py          ← load, validate, remove leakage
        │
        ▼
src/data/feature_engineering.py ← LTI ratio, credit age, purpose groups
        │
        ▼
src/data/preprocessing.py      ← impute, encode, scale, 80/20 split
        │
        ├──► src/models/xgboost_model.py
        ├──► src/models/lightgbm_model.py
        └──► src/models/random_forest_model.py
                    │
                    ▼
        src/evaluation/metrics.py   ← AUC-ROC, F1, Brier, CV
                    │
                    ▼
        src/explainability/shap_analysis.py  ← global + local SHAP
                    │
                    ├──► src/llm/chatbot.py   ← Claude-powered explanations
                    └──► dashboard/app.py     ← Dash UI
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data
cp /path/to/Lending_club_data.csv data/raw/

# 3. Run full pipeline (notebook-based — see notebooks/)
jupyter notebook notebooks/04_model_training.ipynb

# 4. Launch dashboard
python dashboard/app.py
# → http://localhost:8050

# 5. Run tests
pytest tests/
```

## Project Status (Week 5–8 Remaining)

| Task | Status |
|---|---|
| Data Engineering & Preprocessing | Done |
| XGBoost Baseline + Architecture | Done |
| Feature Engineering & EDA | Done |
| Hyperparameter Tuning (HPC) | Active |
| SHAP Analysis (Global + Local) | Active |
| Dashboard UI + SHAP Charts | Active |
| Chatbot + LLM Explainability | Upcoming |
| Macro Stress Testing & Fairness | Upcoming |
| Integration, QA & Final Demo | Upcoming |
