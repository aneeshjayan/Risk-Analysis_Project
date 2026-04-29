"""
main.py — FastAPI backend for the Loan Prediction app.

Endpoints:
  POST /predict         — Soft-ensemble (XGB + LGBM + RF) + SHAP explanation
  POST /chat            — Claude-powered AI advisor with loan context
  GET  /models/metrics  — Comparison table for all trained models
  GET  /health          — Liveness check

Start server:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="LoanLens — Credit Risk API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artefacts ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def _load(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artefact not found: {path}\n"
            "Run  python train_model.py  first."
        )
    return joblib.load(path)

def _load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

try:
    xgb_model    = _load("xgb_model.pkl")
    lgbm_model   = _load("lgbm_model.pkl")
    rf_model     = _load("rf_model.pkl")
    scaler       = _load("scaler.pkl")
    FEATURE_COLS = _load("feature_names.pkl")
    NUMERICAL_COLS = _load("numerical_cols.pkl")
    # XGBoost explainer for SHAP (fastest and most interpretable)
    xgb_explainer = shap.TreeExplainer(xgb_model)
    MODELS_LOADED = True
    print("All model artefacts loaded successfully.")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    xgb_model = lgbm_model = rf_model = scaler = None
    xgb_explainer = FEATURE_COLS = NUMERICAL_COLS = None
    MODELS_LOADED = False

# ── Constants ──────────────────────────────────────────────────────────────────
PURPOSE_GROUP_MAP = {
    "debt_consolidation": "Debt",   "credit_card":      "Debt",
    "home_improvement":   "Asset",  "car":              "Asset",
    "house":              "Asset",  "major_purchase":   "Asset",
    "renewable_energy":   "Asset",  "small_business":   "Business",
    "medical":            "Personal", "vacation":       "Personal",
    "moving":             "Personal", "wedding":        "Personal",
    "educational":        "Personal", "other":          "Other",
}

EMP_MAP = {
    "< 1 year": 0,
    "1 year": 1, "2 years": 2,  "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6,
    "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

FEATURE_LABELS = {
    "loan_amnt":                    "Loan Amount",
    "term":                         "Loan Term (months)",
    "annual_inc":                   "Annual Income",
    "emp_length":                   "Employment Length (yrs)",
    "fico_range_low":               "FICO Score (low)",
    "fico_range_high":              "FICO Score (high)",
    "dti":                          "Debt-to-Income Ratio",
    "inq_last_6mths":               "Credit Inquiries (6 mo)",
    "delinq_2yrs":                  "Late Payments (2 yrs)",
    "acc_now_delinq":               "Accounts Delinquent",
    "collections_12_mths_ex_med":   "Collections (12 mo)",
    "chargeoff_within_12_mths":     "Charge-offs (12 mo)",
    "pub_rec":                      "Public Records",
    "credit_age_years":             "Credit History (yrs)",
    "credit_score":                 "Credit Score",
    "loan_to_income":               "Loan-to-Income Ratio",
    "home_ownership_MORTGAGE":      "Owns (Mortgage)",
    "home_ownership_NONE":          "No Home Ownership",
    "home_ownership_OTHER":         "Other Home Ownership",
    "home_ownership_OWN":           "Owns (Outright)",
    "home_ownership_RENT":          "Rents",
    "application_type_Joint App":   "Joint Application",
    "purpose_group_Business":       "Business Loan",
    "purpose_group_Debt":           "Debt Consolidation",
    "purpose_group_Other":          "Other Purpose",
    "purpose_group_Personal":       "Personal Use",
}


# ── Request / response models ──────────────────────────────────────────────────
class LoanInput(BaseModel):
    loan_amnt:        float = Field(..., gt=0,           description="Loan amount in USD")
    term:             int   = Field(..., ge=36, le=60,   description="36 or 60 months")
    annual_inc:       float = Field(..., gt=0,           description="Annual income in USD")
    emp_length:       str   = Field(...,                 description="e.g. '< 1 year', '5 years', '10+ years'")
    home_ownership:   str   = Field(...,                 description="MORTGAGE | OWN | RENT | NONE | OTHER")
    purpose:          str   = Field(...,                 description="Loan purpose")
    application_type: str   = Field(...,                 description="Individual | Joint App")
    credit_score:     float = Field(..., ge=300, le=850, description="FICO credit score")
    dti:              float = Field(..., ge=0,           description="Debt-to-income ratio (%)")
    credit_age_years: float = Field(..., ge=0,           description="Years of credit history")
    inq_last_6mths:   float = Field(0,  ge=0,            description="Credit inquiries last 6 months")
    delinq_2yrs:      float = Field(0,  ge=0,            description="Late payments last 2 years")


class PredictResponse(BaseModel):
    decision:           str
    probability:        float
    model_probabilities: dict    # individual model probs
    shap_values:        dict
    feature_labels:     dict


class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context:  dict   # prediction result + user inputs + shap values


# ── Preprocessing helper ───────────────────────────────────────────────────────
def preprocess(inp: LoanInput) -> pd.DataFrame:
    d = inp.model_dump()

    d["fico_range_low"]  = d["credit_score"] - 10
    d["fico_range_high"] = d["credit_score"] + 10
    d["loan_to_income"]  = d["loan_amnt"] / d["annual_inc"]
    d["emp_length"]      = EMP_MAP.get(d["emp_length"], 0)

    purpose_group = PURPOSE_GROUP_MAP.get(d["purpose"], "Other")

    for col in ["acc_now_delinq", "collections_12_mths_ex_med",
                "chargeoff_within_12_mths", "pub_rec"]:
        d[col] = 0

    for val in ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]:
        d[f"home_ownership_{val}"] = int(d["home_ownership"].upper() == val)

    d["application_type_Joint App"] = int(d["application_type"] == "Joint App")

    for grp in ["Business", "Debt", "Other", "Personal"]:
        d[f"purpose_group_{grp}"] = int(purpose_group == grp)

    df = pd.DataFrame([d])
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
    return df[FEATURE_COLS]


# ── /predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(inp: LoanInput):
    if not MODELS_LOADED:
        raise HTTPException(status_code=503,
                            detail="Models not loaded. Run train_model.py first.")

    X = preprocess(inp)

    # Individual model probabilities
    xgb_prob  = float(xgb_model.predict_proba(X)[0][1])
    lgbm_prob = float(lgbm_model.predict_proba(X)[0][1])
    rf_prob   = float(rf_model.predict_proba(X)[0][1])

    # Soft ensemble (unweighted average)
    ens_prob = (xgb_prob + lgbm_prob + rf_prob) / 3

    # SHAP from XGBoost (fastest and most reliable for TreeExplainer)
    sv = xgb_explainer.shap_values(X)
    if isinstance(sv, list):
        sv_arr = sv[1][0]
    else:
        sv_arr = sv[0] if sv.ndim == 2 else sv

    shap_dict = {
        feat: round(float(val), 6)
        for feat, val in zip(FEATURE_COLS, sv_arr)
    }

    return PredictResponse(
        decision      = "Rejected" if ens_prob >= 0.5 else "Accepted",
        probability   = round(ens_prob, 4),
        model_probabilities = {
            "xgboost":       round(xgb_prob, 4),
            "lightgbm":      round(lgbm_prob, 4),
            "random_forest": round(rf_prob, 4),
            "ensemble":      round(ens_prob, 4),
        },
        shap_values    = shap_dict,
        feature_labels = FEATURE_LABELS,
    )


# ── /chat endpoint (Claude) ────────────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503,
                            detail="ANTHROPIC_API_KEY not set in .env")

    ctx         = req.context
    decision    = ctx.get("decision", "Unknown")
    probability = ctx.get("probability", 0)
    inputs      = ctx.get("inputs", {})
    shap_values = ctx.get("shap_values", {})
    model_probs = ctx.get("model_probabilities", {})

    # Top-5 SHAP factors
    sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    shap_lines  = "\n".join(
        f"  • {FEATURE_LABELS.get(k, k)}: {'+' if v > 0 else ''}{v:.4f} "
        f"({'increases risk' if v > 0 else 'reduces risk'})"
        for k, v in sorted_shap
    )

    # Model agreement block
    model_lines = "\n".join(
        f"  • {k.replace('_', ' ').title()}: {v*100:.1f}%"
        for k, v in model_probs.items()
    )

    system_prompt = f"""You are a knowledgeable and empathetic financial advisor AI helping users understand their loan application outcome.

## Loan Application Result
- Decision: **{decision}**
- Default Probability (Ensemble): {probability * 100:.1f}%  (≥50% → Rejected)

## Model Agreement
{model_lines}

## Applicant Profile
- Loan Amount: ${inputs.get('loan_amnt', 'N/A'):,.0f}
- Loan Term: {inputs.get('term', 'N/A')} months
- Annual Income: ${inputs.get('annual_inc', 'N/A'):,.0f}
- Employment Length: {inputs.get('emp_length', 'N/A')}
- Home Ownership: {inputs.get('home_ownership', 'N/A')}
- Loan Purpose: {inputs.get('purpose', 'N/A')}
- Application Type: {inputs.get('application_type', 'N/A')}
- Credit Score: {inputs.get('credit_score', 'N/A')}
- Debt-to-Income Ratio: {inputs.get('dti', 'N/A')}%
- Years of Credit History: {inputs.get('credit_age_years', 'N/A')}
- Credit Inquiries (6 mo): {inputs.get('inq_last_6mths', 'N/A')}
- Late Payments (2 yrs): {inputs.get('delinq_2yrs', 'N/A')}

## Top 5 Risk Factors (SHAP)
{shap_lines}

## Your Role
- Explain the decision in plain language (no jargon)
- If rejected: identify main risk factors, give specific actionable advice to improve approval odds
- If accepted: congratulate and highlight the borrower's strengths
- Answer follow-up questions about credit scores, DTI, SHAP factors, or how to improve
- Be empathetic, constructive, and professional
- Keep responses concise (3–5 sentences unless asked for more detail)
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Build message history for Claude (all turns except system)
        messages = [
            {"role": m.role if m.role in ("user", "assistant") else "user",
             "content": m.content}
            for m in req.messages
        ]

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=messages,
        )
        return {"reply": response.content[0].text}

    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ── /models/metrics endpoint ───────────────────────────────────────────────────
@app.get("/models/metrics")
def model_metrics():
    data = _load_json("model_metrics.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="model_metrics.json not found. Run train_model.py first."
        )
    return data


# ── /models/shap endpoint ─────────────────────────────────────────────────────
@app.get("/models/shap")
def model_shap():
    data = _load_json("shap_global.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="shap_global.json not found. Run train_model.py first."
        )
    return data


# ── /health endpoint ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": MODELS_LOADED,
        "models": {
            "xgboost":       xgb_model  is not None,
            "lightgbm":      lgbm_model is not None,
            "random_forest": rf_model   is not None,
        },
    }
