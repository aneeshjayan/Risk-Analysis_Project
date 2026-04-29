/**
 * ShapChart — horizontal bar chart of SHAP values.
 * Positive SHAP (red)  → pushes toward default / rejection
 * Negative SHAP (green)→ pushes toward approval
 * Shows top N features by default; "Show all" toggle reveals every feature.
 */
import { useState } from 'react'

const TOP_N = 12

export default function ShapChart({ shapValues, featureLabels, decision }) {
  const [showAll, setShowAll] = useState(false)

  // Sort ALL features by absolute SHAP value
  const allSorted = Object.entries(shapValues)
    .map(([feat, val]) => ({
      feat,
      label: featureLabels[feat] || feat,
      val:   val,
      abs:   Math.abs(val),
    }))
    .sort((a, b) => b.abs - a.abs)

  const sorted  = showAll ? allSorted : allSorted.slice(0, TOP_N)
  const maxAbs  = allSorted[0]?.abs || 1   // always relative to global max

  return (
    <div className="shap-container">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <div className="shap-title" style={{ marginBottom: 0 }}>
          Feature Impact on Prediction — {showAll ? `All ${allSorted.length}` : `Top ${TOP_N}`} Factors
        </div>
        <button
          onClick={() => setShowAll(v => !v)}
          style={{
            background: 'none', border: '1.5px solid #dde3eb',
            borderRadius: 6, padding: '4px 12px',
            fontSize: 12, fontWeight: 600, cursor: 'pointer',
            color: '#2563eb',
          }}
        >
          {showAll ? `Show Top ${TOP_N} only` : `Show all ${allSorted.length} features`}
        </button>
      </div>

      <p className="text-muted" style={{ marginBottom: 20, fontSize: 13 }}>
        Each bar shows how much a feature pushed the model toward
        <strong style={{ color: '#dc2626' }}> rejection (red)</strong> or
        <strong style={{ color: '#16a34a' }}> approval (green)</strong>.
        Longer bar = stronger influence.
      </p>

      {sorted.map(({ feat, label, val, abs }) => {
        const pct    = (abs / maxAbs) * 100
        const isPos  = val > 0   // positive → increases default risk → bad
        const color  = isPos ? '#ef4444' : '#22c55e'

        return (
          <div className="shap-bar-row" key={feat}>
            <div className="shap-feat-label" title={label}>{label}</div>

            <div className="shap-bar-track">
              <div
                className={`shap-bar-fill ${isPos ? 'shap-bar-pos' : 'shap-bar-neg'}`}
                style={{
                  width: `${pct}%`,
                  left:  0,
                  background: color,
                }}
              />
            </div>

            <div className="shap-val" style={{ color }}>
              {val > 0 ? '+' : ''}{val.toFixed(4)}
            </div>
          </div>
        )
      })}

      {/* Legend */}
      <div className="shap-legend">
        <span>
          <span className="shap-legend-dot" style={{ background: '#ef4444' }} />
          Increases default risk (pushes toward Rejection)
        </span>
        <span>
          <span className="shap-legend-dot" style={{ background: '#22c55e' }} />
          Decreases default risk (pushes toward Approval)
        </span>
      </div>

      <p className="text-muted" style={{ marginTop: 14, fontSize: 12 }}>
        SHAP (SHapley Additive exPlanations) values from the XGBoost model.
        Values are in log-odds units.
      </p>
    </div>
  )
}
