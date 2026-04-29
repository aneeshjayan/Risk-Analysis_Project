/**
 * ScoreGauge — animated SVG ring showing default risk %,
 * plus verdict, risk tier, and model probability pills.
 */

const CIRCUMFERENCE = 2 * Math.PI * 54   // r=54 → ≈ 339.3

function riskColor(prob) {
  if (prob < 0.25) return '#16a34a'
  if (prob < 0.45) return '#f59e0b'
  if (prob < 0.65) return '#ef4444'
  return '#991b1b'
}

function tierClass(tier) {
  if (!tier) return 'tier-high'
  const t = tier.toLowerCase()
  if (t.includes('very')) return 'tier-very-high'
  if (t.includes('high')) return 'tier-high'
  if (t.includes('moderate')) return 'tier-moderate'
  return 'tier-low'
}

const MODEL_COLORS = {
  xgboost:       '#2563eb',
  lightgbm:      '#7c3aed',
  random_forest: '#059669',
}
const MODEL_LABELS = {
  xgboost:       'XGBoost',
  lightgbm:      'LightGBM',
  random_forest: 'Rand. Forest',
}

export default function ScoreGauge({ result, inputs }) {
  const prob      = result.probability
  const accepted  = result.decision === 'Accepted'
  const riskPct   = Math.round(prob * 100)
  const color     = riskColor(prob)
  const offset    = CIRCUMFERENCE * (1 - prob)
  const bestModel = result.model_probabilities?._best

  // Filter out the _best meta key for display
  const modelEntries = Object.entries(result.model_probabilities || {})
    .filter(([k]) => k !== '_best')

  return (
    <div className="score-card">

      {/* ── Gauge ring ──────────────────────────────────────────────────── */}
      <div className="gauge-wrap">
        <svg width="160" height="160" viewBox="0 0 160 160">
          {/* Background ring */}
          <circle className="gauge-bg" cx="80" cy="80" r="54" />
          {/* Risk fill ring */}
          <circle
            className="gauge-fill"
            cx="80" cy="80" r="54"
            stroke={color}
            strokeDasharray={`${CIRCUMFERENCE}`}
            strokeDashoffset={offset}
          />
        </svg>

        {/* Center text */}
        <div className="gauge-center">
          <span className="gauge-pct" style={{ color }}>{riskPct}%</span>
          <span className="gauge-label">default risk</span>
        </div>
      </div>

      {/* ── Verdict ─────────────────────────────────────────────────────── */}
      <div className={`verdict-badge ${accepted ? 'verdict-accepted' : 'verdict-rejected'}`}>
        {accepted ? '✓' : '✕'}&nbsp;{accepted ? 'Likely Approved' : 'Likely Rejected'}
      </div>

      {/* ── Risk tier ───────────────────────────────────────────────────── */}
      <span className={`risk-tier-badge ${tierClass(result.risk_tier)}`}>
        {result.risk_tier}
      </span>

      {/* ── Model probability pills ──────────────────────────────────────── */}
      <div className="model-pills" style={{ width: '100%' }}>
        <div className="section-heading" style={{ marginBottom: 8 }}>
          All Models
        </div>
        {modelEntries.map(([model, p]) => {
          const isBest = model === bestModel
          const col    = MODEL_COLORS[model] ?? '#64748b'
          return (
            <div className="model-pill" key={model}>
              <span className="model-pill-name">
                {MODEL_LABELS[model] ?? model}
              </span>
              <div className="model-pill-bar-track">
                <div
                  className="model-pill-bar-fill"
                  style={{
                    width: `${Math.round(p * 100)}%`,
                    background: col,
                    opacity: isBest ? 1 : 0.55,
                  }}
                />
              </div>
              <span className="model-pill-val" style={{ color: col }}>
                {Math.round(p * 100)}%
              </span>
              {isBest && <span className="best-star">★</span>}
            </div>
          )
        })}
        <p className="text-muted" style={{ fontSize: 11, marginTop: 6 }}>
          ★ = winner (best AUC-ROC) used for prediction
        </p>
      </div>

      {/* ── Application snapshot ────────────────────────────────────────── */}
      {inputs && (
        <div style={{ width: '100%' }}>
          <div className="section-heading" style={{ marginBottom: 8 }}>
            Your Profile
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
            {[
              ['Loan',     `$${Number(inputs.loan_amnt).toLocaleString()}`],
              ['Income',   `$${Number(inputs.annual_inc).toLocaleString()}`],
              ['FICO',     inputs.credit_score],
              ['DTI',      `${inputs.dti}%`],
              ['Term',     `${inputs.term} mo`],
              ['Purpose',  inputs.purpose.replace(/_/g, ' ')],
            ].map(([label, val]) => (
              <div key={label} style={{
                background: '#f8fafc',
                borderRadius: 7,
                padding: '6px 10px',
                fontSize: 12,
              }}>
                <div style={{ color: '#94a3b8', fontWeight: 600, fontSize: 10,
                               textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                  {label}
                </div>
                <div style={{ fontWeight: 700, marginTop: 1, color: '#0f172a',
                               textTransform: 'capitalize' }}>
                  {String(val)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
