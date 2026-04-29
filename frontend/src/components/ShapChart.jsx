/**
 * ShapChart — animated horizontal bar chart of SHAP feature contributions.
 * Red bars  → push toward default / rejection
 * Green bars → push toward approval
 */
import { useState } from 'react'

const TOP_N = 10

export default function ShapChart({ shapValues, featureLabels, decision }) {
  const [showAll, setShowAll] = useState(false)

  const allSorted = Object.entries(shapValues)
    .map(([feat, val]) => ({
      feat,
      label: featureLabels[feat] || feat.replace(/_/g, ' '),
      val,
      abs: Math.abs(val),
    }))
    .sort((a, b) => b.abs - a.abs)

  const displayed = showAll ? allSorted : allSorted.slice(0, TOP_N)
  const maxAbs    = allSorted[0]?.abs || 1

  return (
    <div className="shap-card">
      {/* Header */}
      <div className="shap-header">
        <div>
          <div className="shap-title">Why did the model decide this?</div>
          <div className="shap-subtitle">
            Top {showAll ? allSorted.length : Math.min(TOP_N, allSorted.length)} factors
            driving the prediction — based on SHAP values
          </div>
        </div>
        <button className="show-more-btn" onClick={() => setShowAll(v => !v)}>
          {showAll ? `Top ${TOP_N}` : `All ${allSorted.length}`}
        </button>
      </div>

      {/* Bars */}
      {displayed.map(({ feat, label, val, abs }, i) => {
        const pct   = (abs / maxAbs) * 100
        const isPos = val > 0
        const color = isPos ? '#ef4444' : '#22c55e'
        const bgColor = isPos ? 'rgba(239,68,68,0.08)' : 'rgba(34,197,94,0.08)'

        return (
          <div
            className="shap-bar-row"
            key={feat}
            style={{
              animationDelay: `${i * 0.05}s`,
              borderRadius: 6,
              padding: '3px 4px',
              background: i % 2 === 0 ? 'transparent' : '#fafbfd',
            }}
          >
            <div className="shap-feat-label" title={label}>{label}</div>

            <div className="shap-bar-track">
              <div
                className="shap-bar-fill"
                style={{ width: `${pct}%`, background: color }}
              />
            </div>

            <div className="shap-val" style={{ color }}>
              {val > 0 ? '+' : ''}{val.toFixed(3)}
            </div>
          </div>
        )
      })}

      {/* Legend */}
      <div className="shap-legend">
        <span>
          <span className="shap-legend-dot" style={{ background: '#ef4444' }} />
          Increases default risk → Rejection
        </span>
        <span>
          <span className="shap-legend-dot" style={{ background: '#22c55e' }} />
          Decreases default risk → Approval
        </span>
      </div>

      <p className="text-muted" style={{ marginTop: 12, fontSize: 11.5 }}>
        SHAP (SHapley Additive exPlanations) — each bar shows how much that
        feature moved the prediction away from the average. Longer bar = bigger impact.
      </p>
    </div>
  )
}
