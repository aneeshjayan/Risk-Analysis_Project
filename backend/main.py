"""
main.py — FastAPI backend for the LoanLens Credit Risk app.

Two data sources power this system:
  1. LendingClub loans CSV  — trained the ML models (XGBoost / LightGBM / RF)
  2. FRED macroeconomic API — enriches predictions with current economic context

Endpoints:
  POST /predict        — Predict default risk + SHAP explanation
  POST /chat           — OpenAI GPT-4o-mini financial advisor
  GET  /models/metrics — All model metrics for the dashboard
  GET  /models/shap    — Global SHAP importance for the dashboard
  GET  /health         — Liveness check

Start server:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import datetime
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
from dotenv import load_dotenv

# All LLM logic — system prompt, SHAP context, OpenAI call — lives in src/llm/chatbot.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.llm.chatbot import chat as llm_chat

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── FRED macro — fetched once at server start, cached in memory ────────────────
_MACRO_CACHE: dict = {}

def _load_current_macro() -> dict:
    """Return current macroeconomic snapshot for inference-time feature injection."""
    global _MACRO_CACHE
    if _MACRO_CACHE:
        return _MACRO_CACHE

    fred_key = os.getenv("FRED_API_KEY", "")
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from src.data.macro_features import get_current_macro
        _MACRO_CACHE = get_current_macro(fred_key)
        today = datetime.date.today()
        _MACRO_CACHE["issue_year"]    = today.year
        _MACRO_CACHE["issue_quarter"] = (today.month - 1) // 3 + 1
    except Exception:
        # Safe fallback — server works even without FRED key / fredapi installed
        today = datetime.date.today()
        _MACRO_CACHE = {
            "unemployment_rate":   4.0,
            "fed_funds_rate":      5.33,
            "inflation_cpi":       314.0,
            "real_disposable_inc": 15_800.0,
            "unemp_3m_change":     0.0,
            "recession_flag":      0,
            "issue_year":          today.year,
            "issue_quarter":       (today.month - 1) // 3 + 1,
        }
    return _MACRO_CACHE


# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="LoanLens — Credit Risk API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def _load(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Run  python pipeline.py  first to train the models."
        )
    return joblib.load(path)

def _load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

try:
    # All 3 models loaded — only best is used for predictions, others for the dashboard
    xgb_model      = _load("xgb_model.pkl")
    lgbm_model     = _load("lgbm_model.pkl")
    rf_model       = _load("rf_model.pkl")
    scaler         = _load("scaler.pkl")
    FEATURE_COLS   = _load("feature_names.pkl")
    NUMERICAL_COLS = _load("numerical_cols.pkl")

    # Best model — picked automatically by AUC-ROC during pipeline training
    try:
        BEST_MODEL_NAME = _load("best_model_name.pkl")
    except FileNotFoundError:
        BEST_MODEL_NAME = "xgboost"   # safe fallback

    _ALL_MODELS = {
        "xgboost":       xgb_model,
        "lightgbm":      lgbm_model,
        "random_forest": rf_model,
    }
    best_model     = _ALL_MODELS[BEST_MODEL_NAME]
    shap_explainer = shap.TreeExplainer(best_model)

    MODELS_LOADED = True
    print(f"✓ Models loaded.  Production model → {BEST_MODEL_NAME.upper()} (best AUC)")
    print(f"  Other models loaded for metrics dashboard only.")

except FileNotFoundError as e:
    print(f"WARNING: {e}")
    xgb_model = lgbm_model = rf_model = None
    best_model = shap_explainer = FEATURE_COLS = NUMERICAL_COLS = None
    BEST_MODEL_NAME = "xgboost"
    MODELS_LOADED = False


# ── Constants ──────────────────────────────────────────────────────────────────
PURPOSE_GROUP_MAP = {
    "debt_consolidation": "Debt",    "credit_card":    "Debt",
    "home_improvement":   "Asset",   "car":            "Asset",
    "house":              "Asset",   "major_purchase": "Asset",
    "renewable_energy":   "Asset",   "small_business": "Business",
    "medical":            "Personal","vacation":       "Personal",
    "moving":             "Personal","wedding":        "Personal",
    "educational":        "Personal","other":          "Other",
}

EMP_MAP = {
    "< 1 year": 0,
    "1 year": 1,  "2 years": 2,  "3 years": 3,
    "4 years": 4, "5 years": 5,  "6 years": 6,
    "7 years": 7, "8 years": 8,  "9 years": 9,
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
    # FRED macroeconomic features (Dataset 2)
    "unemployment_rate":            "Unemployment Rate (%)",
    "fed_funds_rate":               "Federal Funds Rate (%)",
    "inflation_cpi":                "CPI Inflation Index",
    "real_disposable_inc":          "Real Disposable Income",
    "unemp_3m_change":              "Unemployment Trend (3mo)",
    "recession_flag":               "Recession Active",
    "issue_year":                   "Loan Issue Year",
    "issue_quarter":                "Loan Issue Quarter",
}


# ── Request / Response models ──────────────────────────────────────────────────
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
    decision:             str    # Accepted / Rejected
    probability:          float  # P(default) from best model
    model_probabilities:  dict   # all 3 model scores — shown in dashboard
    risk_tier:            str    # Low / Moderate / High / Very High Risk
    shap_values:          dict   # feature-level SHAP contributions
    feature_labels:       dict   # human-readable feature names for UI


class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context:  dict   # prediction result + user inputs + shap values


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(inp: LoanInput) -> pd.DataFrame:
    d = inp.model_dump()

    # Derived features
    d["fico_range_low"]  = d["credit_score"] - 10
    d["fico_range_high"] = d["credit_score"] + 10
    d["loan_to_income"]  = d["loan_amnt"] / d["annual_inc"]
    d["emp_length"]      = EMP_MAP.get(d["emp_length"], 0)

    purpose_group = PURPOSE_GROUP_MAP.get(d["purpose"], "Other")

    # Zero-fill infrequent risk flags
    for col in ["acc_now_delinq", "collections_12_mths_ex_med",
                "chargeoff_within_12_mths", "pub_rec"]:
        d[col] = 0

    # One-hot encode categorical fields
    for val in ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]:
        d[f"home_ownership_{val}"] = int(d["home_ownership"].upper() == val)

    d["application_type_Joint App"] = int(d["application_type"] == "Joint App")

    for grp in ["Business", "Debt", "Other", "Personal"]:
        d[f"purpose_group_{grp}"] = int(purpose_group == grp)

    # Inject current FRED macroeconomic context (Dataset 2)
    # At inference time we use today's economic conditions — exactly how
    # production credit scoring models work in practice.
    macro = _load_current_macro()
    for col, val in macro.items():
        d[col] = val

    df = pd.DataFrame([d])

    # Scale only the columns the scaler was actually fitted on.
    # If FRED_API_KEY was absent during training the scaler won't know those cols —
    # passing unknown columns to scaler.transform() raises a ValueError.
    scaler_cols = list(getattr(scaler, 'feature_names_in_', NUMERICAL_COLS))
    num_cols_present = [c for c in scaler_cols if c in df.columns]
    if num_cols_present:
        df[num_cols_present] = scaler.transform(df[num_cols_present])

    # Align to exact training feature set (fills any missing col with 0)
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df


# ── /predict ───────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(inp: LoanInput):
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run  python pipeline.py  first."
        )

    X = preprocess(inp)

    # Best model makes the prediction — picked by AUC during training
    default_prob = float(best_model.predict_proba(X)[0][1])

    # All 3 model scores for the dashboard comparison panel
    xgb_prob  = float(xgb_model.predict_proba(X)[0][1])
    lgbm_prob = float(lgbm_model.predict_proba(X)[0][1])
    rf_prob   = float(rf_model.predict_proba(X)[0][1])

    decision  = "Rejected" if default_prob >= 0.5 else "Accepted"
    risk_tier = (
        "Low Risk"       if default_prob < 0.25 else
        "Moderate Risk"  if default_prob < 0.45 else
        "High Risk"      if default_prob < 0.65 else
        "Very High Risk"
    )

    # SHAP — explains WHY the best model made this prediction
    sv = shap_explainer.shap_values(X)
    if isinstance(sv, list):
        sv_arr = sv[1][0]
    elif sv.ndim == 2:
        sv_arr = sv[0]
    else:
        sv_arr = sv

    shap_dict = {
        feat: round(float(val), 6)
        for feat, val in zip(FEATURE_COLS, sv_arr)
    }

    return PredictResponse(
        decision            = decision,
        probability         = round(default_prob, 4),
        model_probabilities = {
            "xgboost":       round(xgb_prob, 4),
            "lightgbm":      round(lgbm_prob, 4),
            "random_forest": round(rf_prob, 4),
            "_best":         BEST_MODEL_NAME,
        },
        risk_tier     = risk_tier,
        shap_values   = shap_dict,
        feature_labels= FEATURE_LABELS,
    )


# ── /chat ──────────────────────────────────────────────────────────────────────
# All LLM logic lives in src/llm/chatbot.py — backend just calls it.
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    ctx = req.context
    try:
        reply = llm_chat(
            messages    = [m.model_dump() for m in req.messages],
            decision    = ctx.get("decision", "Unknown"),
            probability = ctx.get("probability", 0),
            risk_tier   = ctx.get("risk_tier", "Unknown"),
            best_model  = BEST_MODEL_NAME,
            model_probs = ctx.get("model_probabilities", {}),
            inputs      = ctx.get("inputs", {}),
            shap_values = ctx.get("shap_values", {}),
            macro       = _load_current_macro(),
            api_key     = os.getenv("OPENAI_API_KEY", ""),
        )
        return {"reply": reply}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ── /macro ─────────────────────────────────────────────────────────────────────
@app.get("/macro")
def macro_conditions():
    """
    Return current FRED macroeconomic snapshot with plain-English interpretation.
    Used by the frontend MacroWidget and passed as context to the LLM copilot.
    """
    m = _load_current_macro()

    # Derive human-readable signals from the raw values
    unemp      = m.get("unemployment_rate",   4.0)
    fed_rate   = m.get("fed_funds_rate",      5.33)
    recession  = bool(m.get("recession_flag", 0))
    unemp_chg  = m.get("unemp_3m_change",     0.0)
    cpi        = m.get("inflation_cpi",        314.0)
    income     = m.get("real_disposable_inc",  15800.0)

    def _trend(val):
        if val >  0.1: return "rising"
        if val < -0.1: return "falling"
        return "stable"

    # Lending environment assessment
    if recession:
        climate = "Recession"
        climate_color = "red"
        climate_msg   = "Lenders are tightening standards during a recession. Approvals are harder to get."
    elif fed_rate >= 5.0:
        climate = "High Rate Environment"
        climate_color = "orange"
        climate_msg   = f"The Fed rate is {fed_rate}% — borrowing costs are elevated. Monthly payments are higher than usual."
    elif unemp >= 5.5 or _trend(unemp_chg) == "rising":
        climate = "Cautious Market"
        climate_color = "orange"
        climate_msg   = "Unemployment is rising — lenders are becoming more selective about credit risk."
    else:
        climate = "Stable Market"
        climate_color = "green"
        climate_msg   = "Economic conditions are relatively stable. Normal lending standards apply."

    return {
        "indicators": {
            "unemployment_rate":  {
                "value":  round(unemp, 2),
                "unit":   "%",
                "label":  "Unemployment Rate",
                "trend":  _trend(unemp_chg),
                "change": round(unemp_chg, 2),
                "status": "bad" if unemp >= 6 else "warn" if unemp >= 4.5 else "good",
                "note":   "Higher unemployment → tighter lending standards",
            },
            "fed_funds_rate": {
                "value":  round(fed_rate, 2),
                "unit":   "%",
                "label":  "Federal Funds Rate",
                "trend":  "stable",
                "change": 0,
                "status": "bad" if fed_rate >= 5 else "warn" if fed_rate >= 3 else "good",
                "note":   "Higher rate = more expensive borrowing for everyone",
            },
            "inflation_cpi": {
                "value":  round(cpi, 1),
                "unit":   "index",
                "label":  "CPI Inflation Index",
                "trend":  "stable",
                "change": 0,
                "status": "warn",
                "note":   "Inflation erodes purchasing power and tightens real incomes",
            },
            "real_disposable_inc": {
                "value":  round(income, 0),
                "unit":   "$/yr",
                "label":  "Real Disposable Income",
                "trend":  "stable",
                "change": 0,
                "status": "good" if income >= 15000 else "warn",
                "note":   "Higher disposable income → better repayment capacity across all borrowers",
            },
            "recession_flag": {
                "value":  1 if recession else 0,
                "unit":   "",
                "label":  "Recession Active",
                "trend":  "stable",
                "change": 0,
                "status": "bad" if recession else "good",
                "note":   "Recessions cause lenders to significantly tighten credit criteria",
            },
        },
        "climate": {
            "label":   climate,
            "color":   climate_color,
            "message": climate_msg,
        },
        "impact_on_loan": _loan_impact(unemp, fed_rate, recession, unemp_chg),
    }


def _loan_impact(unemp: float, fed_rate: float, recession: bool, unemp_chg: float) -> list[str]:
    """Generate bullet-point impacts of current macro on a loan application."""
    impacts = []
    if recession:
        impacts.append("🔴 Recession active — lenders are rejecting borderline applications at higher rates than usual")
    if fed_rate >= 5.0:
        impacts.append(f"🔴 Fed rate at {fed_rate}% — your monthly payment will be significantly higher than in low-rate periods")
    if unemp_chg > 0.2:
        impacts.append(f"🟡 Unemployment rising ({unemp_chg:+.2f}pp) — lenders are becoming more cautious about future defaults")
    if unemp >= 5.5:
        impacts.append(f"🟡 Unemployment at {unemp}% — economic stress is elevated, tighter approval criteria expected")
    if not impacts:
        impacts.append("🟢 Economic conditions are relatively stable — standard approval criteria apply")
    return impacts


# ── /models/metrics ────────────────────────────────────────────────────────────
@app.get("/models/metrics")
def model_metrics():
    data = _load_json("model_metrics.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="model_metrics.json not found — run python pipeline.py first."
        )
    return data


# ── /models/shap ───────────────────────────────────────────────────────────────
@app.get("/models/shap")
def model_shap():
    data = _load_json("shap_global.json")
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="shap_global.json not found — run python pipeline.py first."
        )
    return data


# ── /health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "models_loaded":    MODELS_LOADED,
        "production_model": BEST_MODEL_NAME,
        "datasets": {
            "dataset_1": "LendingClub accepted loans (ML training data)",
            "dataset_2": "FRED macroeconomic API (inference-time enrichment)",
        },
        "models_available": {
            "xgboost":       xgb_model  is not None,
            "lightgbm":      lgbm_model is not None,
            "random_forest": rf_model   is not None,
        },
    }
