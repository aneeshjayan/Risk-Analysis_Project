/**
 * api.ts — Real backend integration for LoanLens.
 *
 * Transforms data between the Lovable frontend types and the FastAPI backend format.
 *
 * Backend:  POST /predict  POST /chat  GET /macro  GET /models/metrics
 * Backend lives at localhost:8000 — proxied by Vite in dev.
 */

// ── Types ──────────────────────────────────────────────────────────────────────

export type LoanInput = {
  loan_amnt:                  number;
  term:                       36 | 60;
  annual_inc:                 number;
  emp_length:                 number;           // 0-10 (slider)
  home_ownership:             "RENT" | "OWN" | "MORTGAGE";
  purpose:                    string;
  application_type:           "Individual" | "Joint";
  fico_range_low:             number;
  fico_range_high:            number;
  dti:                        number;
  inq_last_6mths:             number;
  delinq_2yrs:                number;
  acc_now_delinq:             number;
  collections_12_mths_ex_med: number;
  chargeoff_within_12_mths:   number;
  pub_rec:                    number;
  earliest_cr_line:           string;           // YYYY-MM
};

export type ShapValue  = { feature: string; value: string | number; shap: number };
export type ModelKey   = "XGBoost" | "LightGBM" | "RandomForest";

export type PredictResponse = {
  decision:       "APPROVED" | "REJECTED";
  probability:    number;                        // P(approval) = 1 - P(default)
  risk_tier:      "Low" | "Moderate" | "High" | "Very High";
  best_model:     ModelKey;
  model_probs:    Record<ModelKey, number>;
  shap_values:    ShapValue[];
  feature_labels: string[];
  // raw backend values kept for chat context
  _raw_default_prob: number;
  _raw_model_probs:  Record<string, number>;
  _raw_shap_dict:    Record<string, number>;
};

export type ChatMessage = { role: "user" | "assistant" | "system"; content: string };
export type ChatContext = {
  decision?:    PredictResponse["decision"];
  probability?: number;
  risk_tier?:   PredictResponse["risk_tier"];
  best_model?:  ModelKey;
  model_probs?: Record<ModelKey, number>;
  inputs?:      LoanInput;
  shap_values?: ShapValue[];
  // raw values for backend LLM context
  _raw_default_prob?: number;
  _raw_model_probs?:  Record<string, number>;
  _raw_shap_dict?:    Record<string, number>;
};

export type Indicator = {
  label: string;
  value: string;
  trend: "up" | "down" | "flat";
  tone:  "good" | "warn" | "bad";
  note:  string;
};

export type MacroResponse = {
  indicators: {
    unemployment:          Indicator;
    fed_funds:             Indicator;
    cpi:                   Indicator;
    real_disposable_income: Indicator;
    recession_active:      Indicator;
  };
  climate: { color: "green" | "amber" | "red"; label: string; emoji: string; message: string };
  impact_on_loan: string[];
};

export type ModelMetrics = {
  model:      ModelKey;
  auc_roc:    number;
  f1:         number;
  precision:  number;
  recall:     number;
  brier:      number;
  is_winner?: boolean;
};


// ── Helpers ────────────────────────────────────────────────────────────────────

/** Convert slider number (0-10) to backend string format */
function empLengthStr(n: number): string {
  if (n === 0)  return "< 1 year";
  if (n >= 10)  return "10+ years";
  return `${n} year${n === 1 ? "" : "s"}`;
}

/** Compute credit_age_years from "YYYY-MM" string */
function creditAgeYears(earliest: string): number {
  const [yr, mo] = earliest.split("-").map(Number);
  const msPerYear = 365.25 * 24 * 3600 * 1000;
  return (Date.now() - new Date(yr, mo - 1).getTime()) / msPerYear;
}

const MODEL_KEY_MAP: Record<string, ModelKey> = {
  xgboost:       "XGBoost",
  lightgbm:      "LightGBM",
  random_forest: "RandomForest",
};

const TREND_MAP: Record<string, Indicator["trend"]> = {
  rising:  "up",
  falling: "down",
  stable:  "flat",
};

const TONE_MAP: Record<string, Indicator["tone"]> = {
  good: "good",
  warn: "warn",
  bad:  "bad",
};

const CLIMATE_COLOR_MAP: Record<string, MacroResponse["climate"]["color"]> = {
  green:  "green",
  orange: "amber",
  red:    "red",
};

const CLIMATE_EMOJI: Record<string, string> = {
  green:  "🌤",
  orange: "⚠️",
  red:    "🔴",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function transformIndicator(raw: any, valueOverride?: (v: unknown) => string): Indicator {
  const val = valueOverride
    ? valueOverride(raw.value)
    : raw.unit === "%"
      ? `${raw.value}%`
      : String(raw.value);
  return {
    label: raw.label,
    value: val,
    trend: TREND_MAP[raw.trend] ?? "flat",
    tone:  TONE_MAP[raw.status] ?? "good",
    note:  raw.note,
  };
}


// ── /predict ───────────────────────────────────────────────────────────────────

export async function predict(input: LoanInput): Promise<PredictResponse> {
  // Transform Lovable form values → backend LoanInput format
  const backendBody = {
    loan_amnt:        input.loan_amnt,
    term:             input.term,
    annual_inc:       input.annual_inc,
    emp_length:       empLengthStr(input.emp_length),
    home_ownership:   input.home_ownership,
    purpose:          input.purpose,
    application_type: input.application_type === "Joint" ? "Joint App" : "Individual",
    credit_score:     (input.fico_range_low + input.fico_range_high) / 2,
    dti:              input.dti,
    credit_age_years: creditAgeYears(input.earliest_cr_line),
    inq_last_6mths:   input.inq_last_6mths,
    delinq_2yrs:      input.delinq_2yrs,
  };

  const res = await fetch("/predict", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(backendBody),
  });

  if (!res.ok) {
    let msg = `Server error ${res.status}`;
    try {
      const detail = await res.json();
      msg = detail.detail || JSON.stringify(detail) || msg;
    } catch {
      msg = (await res.text().catch(() => msg)) || msg;
    }
    throw new Error(msg);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();

  // decision: "Accepted"/"Rejected" → "APPROVED"/"REJECTED"
  const decision: PredictResponse["decision"] =
    raw.decision === "Accepted" ? "APPROVED" : "REJECTED";

  // probability: backend P(default) → P(approval) = 1 - p
  const defaultProb: number = raw.probability;
  const approvalProb = 1 - defaultProb;

  // risk_tier: strip " Risk" suffix
  const risk_tier = (raw.risk_tier as string).replace(
    " Risk", "",
  ) as PredictResponse["risk_tier"];

  // best_model from model_probabilities._best
  const bestRaw: string = raw.model_probabilities._best ?? "xgboost";
  const best_model: ModelKey = MODEL_KEY_MAP[bestRaw] ?? "XGBoost";

  // model_probs: lowercase keys → PascalCase ModelKey
  const rawProbs: Record<string, number> = raw.model_probabilities;
  const model_probs: Record<ModelKey, number> = {
    XGBoost:      rawProbs.xgboost       ?? 0,
    LightGBM:     rawProbs.lightgbm      ?? 0,
    RandomForest: rawProbs.random_forest ?? 0,
  };

  // shap_values: dict → ShapValue[] sorted by |shap|, top 10
  const rawShap: Record<string, number> = raw.shap_values ?? {};
  const rawLabels: Record<string, string> = raw.feature_labels ?? {};

  const shap_values: ShapValue[] = Object.entries(rawShap)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 10)
    .map(([feat, shap]) => ({
      feature: rawLabels[feat] ?? feat.replace(/_/g, " "),
      value:   "",
      shap,
    }));

  return {
    decision,
    probability:   approvalProb,
    risk_tier,
    best_model,
    model_probs,
    shap_values,
    feature_labels: shap_values.map((s) => s.feature),
    // keep raw backend values for LLM chat context
    _raw_default_prob: defaultProb,
    _raw_model_probs: {
      xgboost:       rawProbs.xgboost       ?? 0,
      lightgbm:      rawProbs.lightgbm      ?? 0,
      random_forest: rawProbs.random_forest ?? 0,
    },
    _raw_shap_dict: rawShap,
  };
}


// ── /chat ──────────────────────────────────────────────────────────────────────

export async function chat(
  messages: ChatMessage[],
  context:  ChatContext = {},
): Promise<{ reply: string }> {
  // Build backend-compatible inputs from the Lovable form input
  const inp = context.inputs;
  const backendInputs = inp
    ? {
        loan_amnt:        inp.loan_amnt,
        term:             inp.term,
        annual_inc:       inp.annual_inc,
        emp_length:       empLengthStr(inp.emp_length),
        home_ownership:   inp.home_ownership,
        purpose:          inp.purpose,
        application_type: inp.application_type === "Joint" ? "Joint App" : "Individual",
        credit_score:     (inp.fico_range_low + inp.fico_range_high) / 2,
        dti:              inp.dti,
        credit_age_years: creditAgeYears(inp.earliest_cr_line),
        inq_last_6mths:   inp.inq_last_6mths,
        delinq_2yrs:      inp.delinq_2yrs,
      }
    : {};

  const res = await fetch("/chat", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map(({ role, content }) => ({ role, content })),
      context: {
        // Pass backend-format values so the GPT system prompt is accurate
        decision:           context.decision === "APPROVED" ? "Accepted" : "Rejected",
        probability:        context._raw_default_prob ?? (1 - (context.probability ?? 0)),
        risk_tier:          context.risk_tier ? `${context.risk_tier} Risk` : "Unknown",
        model_probabilities: context._raw_model_probs ?? {},
        shap_values:        context._raw_shap_dict ?? {},
        inputs:             backendInputs,
      },
    }),
  });

  if (!res.ok) {
    let msg = `Chat error ${res.status}`;
    try {
      const d = await res.json();
      msg = d.detail || msg;
    } catch { /* ignore */ }
    throw new Error(msg);
  }

  return res.json() as Promise<{ reply: string }>;
}


// ── /macro ─────────────────────────────────────────────────────────────────────

export async function getMacro(): Promise<MacroResponse> {
  const res = await fetch("/macro");
  if (!res.ok) throw new Error(`Macro fetch failed: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();

  const rawInd = raw.indicators ?? {};

  return {
    indicators: {
      unemployment: transformIndicator(rawInd.unemployment_rate),
      fed_funds:    transformIndicator(rawInd.fed_funds_rate),
      cpi:          transformIndicator(rawInd.inflation_cpi),
      real_disposable_income: transformIndicator(
        rawInd.real_disposable_inc,
        (v) => `$${Number(v).toLocaleString()}`,
      ),
      recession_active: transformIndicator(
        rawInd.recession_flag,
        (v) => (v ? "Yes" : "No"),
      ),
    },
    climate: {
      color:   CLIMATE_COLOR_MAP[raw.climate?.color] ?? "green",
      label:   raw.climate?.label   ?? "Stable Market",
      emoji:   CLIMATE_EMOJI[raw.climate?.color]  ?? "🌤",
      message: raw.climate?.message ?? "",
    },
    impact_on_loan: raw.impact_on_loan ?? [],
  };
}


// ── /models/metrics ────────────────────────────────────────────────────────────

export async function getModelMetrics(): Promise<ModelMetrics[]> {
  const res = await fetch("/models/metrics");
  if (!res.ok) throw new Error(`Metrics fetch failed: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();

  // Backend returns { xgboost: {...}, lightgbm: {...}, random_forest: {...} }
  const models: ModelMetrics[] = (
    [
      { model: "XGBoost"      as ModelKey, ...(raw.xgboost       ?? {}) },
      { model: "LightGBM"     as ModelKey, ...(raw.lightgbm      ?? {}) },
      { model: "RandomForest" as ModelKey, ...(raw.random_forest  ?? {}) },
    ] as ModelMetrics[]
  ).filter((m) => m.auc_roc !== undefined);

  const winner = models.reduce(
    (a, b) => (a.auc_roc >= b.auc_roc ? a : b),
    models[0],
  );
  return models.map((m) => ({ ...m, is_winner: m.model === winner?.model }));
}
