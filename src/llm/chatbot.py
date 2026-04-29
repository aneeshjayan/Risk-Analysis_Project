"""
chatbot.py — LLM-powered copilot for explainable credit risk decisions.

Uses OpenAI GPT-4o-mini. Receives the full prediction context:
  - Loan application inputs (12 features)
  - ML model decision + probability + risk tier
  - SHAP values for every feature (top 10 surfaced to LLM)
  - Current FRED macroeconomic snapshot

The LLM uses SHAP values to explain exactly WHY the model decided
what it did, and gives specific, actionable improvement advice.

Owner: Subramanian Raj Narayanan
"""

import os
from typing import Optional
from openai import OpenAI

# ── Feature display labels ────────────────────────────────────────────────────
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
    "home_ownership_MORTGAGE":      "Home: Mortgage",
    "home_ownership_OWN":           "Home: Own Outright",
    "home_ownership_RENT":          "Home: Renting",
    "home_ownership_NONE":          "Home: None",
    "application_type_Joint App":   "Joint Application",
    "purpose_group_Business":       "Purpose: Business",
    "purpose_group_Debt":           "Purpose: Debt Consolidation",
    "purpose_group_Other":          "Purpose: Other",
    "purpose_group_Personal":       "Purpose: Personal",
    "unemployment_rate":            "Unemployment Rate (%)",
    "fed_funds_rate":               "Federal Funds Rate (%)",
    "inflation_cpi":                "CPI Inflation",
    "real_disposable_inc":          "Real Disposable Income",
    "unemp_3m_change":              "Unemployment Trend (3mo)",
    "recession_flag":               "Recession Active",
    "issue_year":                   "Loan Issue Year",
    "issue_quarter":                "Loan Issue Quarter",
}


def _fmt_input_val(key: str, inputs: dict) -> str:
    """Return a human-readable value for a feature key from the raw inputs dict."""
    if key in inputs:
        v = inputs[key]
        if key in ("loan_amnt", "annual_inc"):
            return f"${float(v):,.0f}"
        if key == "dti":
            return f"{v}%"
        return str(v)
    if key.startswith("home_ownership_"):
        return inputs.get("home_ownership", "")
    if key.startswith("purpose_group_"):
        return inputs.get("purpose", "").replace("_", " ")
    if key.startswith("application_type_"):
        return inputs.get("application_type", "")
    return "N/A"


def _macro_interpretation(macro: dict) -> str:
    """
    Convert raw FRED numbers into plain-English bullet points for the LLM.
    Tells GPT what the macro environment means for this specific loan applicant.
    """
    lines = []
    fed   = float(macro.get("fed_funds_rate",    0))
    unemp = float(macro.get("unemployment_rate", 0))
    rec   = bool(macro.get("recession_flag",     0))
    trend = float(macro.get("unemp_3m_change",   0))

    if rec:
        lines.append("• RECESSION: Lenders are rejecting borderline cases at far higher rates. Applicant needs very strong credit to overcome this.")
    if fed >= 5.0:
        lines.append(f"• HIGH RATES ({fed}%): Monthly payments are significantly elevated. A $15K 36-month loan costs ~$450/mo vs ~$430/mo in a 3% rate environment. Mention this impact.")
    elif fed >= 3.0:
        lines.append(f"• MODERATE RATES ({fed}%): Borrowing costs are somewhat elevated but manageable.")
    else:
        lines.append(f"• LOW RATES ({fed}%): Good time to borrow — monthly payments are near historical lows.")

    if unemp >= 6.0:
        lines.append(f"• HIGH UNEMPLOYMENT ({unemp}%): Lenders expect more defaults — approval criteria are stricter than usual.")
    elif unemp >= 4.5 or trend > 0.2:
        lines.append(f"• RISING UNEMPLOYMENT ({unemp}%, {trend:+.2f}pp): Lenders are becoming cautious. Stable employment history matters more now.")
    else:
        lines.append(f"• STABLE EMPLOYMENT ({unemp}%): Labour market is healthy — no extra pressure on lending standards from this factor.")

    return "\n".join(lines)


def build_system_prompt(
    decision:     str,
    probability:  float,
    risk_tier:    str,
    best_model:   str,
    model_probs:  dict,
    inputs:       dict,
    shap_values:  dict,
    macro:        dict,
) -> str:
    """
    Build the full GPT system prompt with prediction result, applicant profile,
    SHAP top-10 factors (with actual input values), and macro context.

    This is the single source of truth for what the LLM knows — all context
    is injected here so responses are grounded in real data, not hallucinated.
    """

    # ── Top-10 SHAP factors with actual input values ──────────────────────────
    top_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    shap_lines = "\n".join(
        f"  {'🔴' if v > 0 else '🟢'} "
        f"{FEATURE_LABELS.get(k, k.replace('_', ' '))}"
        f"  [applicant value: {_fmt_input_val(k, inputs)}]"
        f"  SHAP: {v:+.4f}"
        f"  ({'↑ pushes toward default/rejection' if v > 0 else '↓ pushes toward approval'})"
        for k, v in top_shap
    )

    # ── Model comparison lines ────────────────────────────────────────────────
    model_lines = "\n".join(
        f"  • {k.replace('_', ' ').title()}"
        f"{'  ★ decision model' if k == best_model else ''}"
        f": {v * 100:.1f}% default probability"
        for k, v in model_probs.items()
        if k != "_best"
    )

    return f"""You are LoanLens Copilot — a knowledgeable, empathetic AI financial advisor.
You have complete access to the applicant's loan data, the ML model's prediction,
SHAP explainability scores, and current macroeconomic conditions.
Use all of this to give precise, personalised, actionable advice.

━━━ LOAN DECISION ━━━
Decision:            {decision}
Risk Tier:           {risk_tier}
Default Probability: {probability * 100:.1f}%   (≥50% → Rejected)
Production Model:    {best_model.upper()} — winner by AUC-ROC across 3 trained models

━━━ ALL 3 MODEL SCORES ━━━
{model_lines}
(XGBoost, LightGBM, Random Forest all trained on LendingClub data. Best AUC used for decision.)

━━━ APPLICANT PROFILE ━━━
Loan Amount:         ${inputs.get('loan_amnt', 'N/A'):,.0f}
Loan Term:           {inputs.get('term', 'N/A')} months
Annual Income:       ${inputs.get('annual_inc', 'N/A'):,.0f}
Employment Length:   {inputs.get('emp_length', 'N/A')}
Home Ownership:      {inputs.get('home_ownership', 'N/A')}
Loan Purpose:        {inputs.get('purpose', 'N/A')}
Application Type:    {inputs.get('application_type', 'N/A')}
Credit Score (FICO): {inputs.get('credit_score', 'N/A')}
Debt-to-Income:      {inputs.get('dti', 'N/A')}%
Credit History:      {inputs.get('credit_age_years', 'N/A')} years
Credit Inquiries:    {inputs.get('inq_last_6mths', 'N/A')} (last 6 months)
Late Payments:       {inputs.get('delinq_2yrs', 'N/A')} (last 2 years)

━━━ SHAP EXPLAINABILITY — TOP 10 FACTORS ━━━
SHAP (SHapley Additive exPlanations) shows exactly how much each feature
moved the model toward or away from predicting a default.
Positive = increases default risk. Negative = reduces default risk.
Larger absolute value = stronger influence on this decision.

{shap_lines}

━━━ CURRENT ECONOMIC ENVIRONMENT (FRED — Federal Reserve Data) ━━━
Unemployment Rate:   {macro.get('unemployment_rate', 'N/A')}%  {'⚠ elevated' if float(macro.get('unemployment_rate',0)) >= 5.5 else '✓ normal'}
Federal Funds Rate:  {macro.get('fed_funds_rate', 'N/A')}%  {'⚠ high — expensive borrowing' if float(macro.get('fed_funds_rate',0)) >= 5 else '✓ moderate'}
CPI Inflation:       {macro.get('inflation_cpi', 'N/A')}
Real Disposable Inc: ${float(macro.get('real_disposable_inc', 0)):,.0f}/yr
Recession Active:    {'YES ⚠ — lenders are significantly tightening standards' if macro.get('recession_flag', 0) else 'No — normal lending environment'}
Unemployment Trend:  {macro.get('unemp_3m_change', 0):+.2f}pp over last 3 months  {'(rising ⚠)' if float(macro.get('unemp_3m_change',0)) > 0.1 else '(stable)' if abs(float(macro.get('unemp_3m_change',0))) <= 0.1 else '(falling ✓)'}

MACRO INTERPRETATION FOR THIS APPLICANT:
{_macro_interpretation(macro)}

━━━ YOUR INSTRUCTIONS ━━━
• Use SHAP factors to explain the decision — always reference the actual applicant value and SHAP score
• Speak in plain English — no technical jargon, no raw numbers without context
• If REJECTED:
    - Name the top 2–3 SHAP drivers by name and value
    - Give specific, numerical improvement targets
      (e.g. "Raise credit score from 620 → 680", "Lower DTI from 28% → below 20%")
    - Mention how long realistic improvements take
• If ACCEPTED:
    - Highlight the strongest positive SHAP factors
    - Mention any weak spots to watch
• Reference macro environment where relevant (high rates = higher monthly cost, recession = tighter standards)
• Answer follow-up questions using the data above — never guess or hallucinate numbers
• Be warm, encouraging, and constructive — never judgmental
• Keep responses to 3–5 sentences unless the user explicitly asks for more detail
"""


def build_explanation_prompt(
    features:    dict,
    shap_values: dict,
    predicted_pd: float,
) -> str:
    """
    Build a one-shot explanation prompt (no chat history).
    Used for generating an automatic summary without a conversation.
    """
    top_pos = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)[:5]
    top_neg = sorted(shap_values.items(), key=lambda x: x[1])[:5]

    return (
        f"Predicted Default Probability: {predicted_pd:.2%}\n\n"
        "Top risk-INCREASING factors:\n"
        + "\n".join(
            f"  - {FEATURE_LABELS.get(k, k)}: SHAP={v:+.4f}, value={features.get(k, 'N/A')}"
            for k, v in top_pos
        )
        + "\n\nTop risk-REDUCING factors:\n"
        + "\n".join(
            f"  - {FEATURE_LABELS.get(k, k)}: SHAP={v:+.4f}, value={features.get(k, 'N/A')}"
            for k, v in top_neg
        )
        + "\n\nExplain this prediction in 3 plain-English sentences."
    )


def chat(
    messages:    list[dict],
    decision:    str,
    probability: float,
    risk_tier:   str,
    best_model:  str,
    model_probs: dict,
    inputs:      dict,
    shap_values: dict,
    macro:       dict,
    api_key:     Optional[str] = None,
    model:       str = "gpt-4o-mini",
    max_tokens:  int = 512,
) -> str:
    """
    Send a chat turn to GPT with full loan + SHAP context injected as system prompt.

    Args:
        messages:    Full conversation history [{role, content}, ...]
        decision:    "Accepted" or "Rejected"
        probability: P(default) from the ML model
        risk_tier:   "Low Risk" / "Moderate Risk" / "High Risk" / "Very High Risk"
        best_model:  Name of the winning model (e.g. "xgboost")
        model_probs: {model_name: probability} for all 3 models
        inputs:      Raw loan application inputs from the user
        shap_values: {feature_name: shap_value} for all features
        macro:       Current FRED macroeconomic snapshot
        api_key:     OpenAI API key (falls back to OPENAI_API_KEY env var)
        model:       OpenAI model name
        max_tokens:  Max response length

    Returns:
        The assistant's reply string.
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set. Add it to backend/.env")

    system_prompt = build_system_prompt(
        decision=decision,
        probability=probability,
        risk_tier=risk_tier,
        best_model=best_model,
        model_probs=model_probs,
        inputs=inputs,
        shap_values=shap_values,
        macro=macro,
    )

    client = OpenAI(api_key=key)

    full_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        full_messages.append({"role": role, "content": m.get("content", "")})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        messages=full_messages,
    )
    return response.choices[0].message.content


def explain_prediction(
    features:    dict,
    shap_values: dict,
    predicted_pd: float,
    api_key:     Optional[str] = None,
    model:       str = "gpt-4o-mini",
) -> str:
    """
    One-shot: generate a plain-language explanation for a single prediction.
    No chat history — just returns a concise explanation string.
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=key)
    prompt = build_explanation_prompt(features, shap_values, predicted_pd)

    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are a concise credit risk explanation assistant."},
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content
