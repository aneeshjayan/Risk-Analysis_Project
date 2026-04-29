/**
 * ModelComparison — fetches /models/metrics and renders a visual bar
 * comparison + full table for all trained models.
 */
import { useState, useEffect } from 'react'

const METRIC_LABELS = {
  roc_auc:   'AUC-ROC',
  f1:        'F1 Score',
  precision: 'Precision',
  recall:    'Recall',
  brier_score: 'Brier Score',
}

const MODEL_COLORS = {
  xgboost:       '#2563eb',
  lightgbm:      '#7c3aed',
  random_forest: '#059669',
}
const MODEL_LABELS = {
  xgboost:       'XGBoost',
  lightgbm:      'LightGBM',
  random_forest: 'Random Forest',
}

export default function ModelComparison() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [focus,   setFocus]   = useState('roc_auc')

  useEffect(() => {
    fetch('/models/metrics')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(data => { setMetrics(data); setLoading(false) })
      .catch(e  => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <p style={{ padding: 20, color: '#64748b' }}>Loading model metrics…</p>
  if (error)   return (
    <div style={{ padding: 20, color: '#dc2626' }}>
      <strong>Could not load metrics.</strong> Run <code>python pipeline.py</code> first.
      <br /><small>{error}</small>
    </div>
  )

  // Only keep the three model keys — ignore any meta keys
  const modelKeys = Object.keys(metrics).filter(k =>
    ['xgboost', 'lightgbm', 'random_forest'].includes(k)
  )

  const lowerIsBetter = focus === 'brier_score'
  const focusVals     = modelKeys.map(k => metrics[k]?.[focus] ?? 0)
  const focusMax      = Math.max(...focusVals)
  const bestByFocus   = lowerIsBetter
    ? modelKeys[focusVals.indexOf(Math.min(...focusVals))]
    : modelKeys[focusVals.indexOf(focusMax)]

  // Best overall by AUC
  const bestAUC = modelKeys.reduce((a, b) =>
    (metrics[a]?.roc_auc ?? 0) > (metrics[b]?.roc_auc ?? 0) ? a : b
  )

  return (
    <div>
      <div className="section-heading">Model Performance Comparison</div>
      <p className="text-muted" style={{ marginBottom: 20 }}>
        All three models trained on 80% of LendingClub data, evaluated on a 20% holdout test set.
        ★ marks the winner (best AUC-ROC) used for production predictions.
      </p>

      {/* Metric selector */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
        {Object.entries(METRIC_LABELS).map(([key, label]) => (
          <button key={key} onClick={() => setFocus(key)} style={{
            padding:    '5px 14px',
            borderRadius: 7,
            border:     `1.5px solid ${focus === key ? '#2563eb' : '#dde3eb'}`,
            background: focus === key ? '#2563eb' : '#fff',
            color:      focus === key ? '#fff'    : '#374151',
            fontWeight: 600,
            fontSize:   13,
            cursor:     'pointer',
            fontFamily: 'inherit',
          }}>
            {label}
          </button>
        ))}
      </div>

      {/* Bar comparison */}
      <div style={{ marginBottom: 28 }}>
        {modelKeys.map(key => {
          const val   = metrics[key]?.[focus] ?? 0
          const pct   = lowerIsBetter
            ? focusMax > 0 ? (1 - val / focusMax) * 100 : 0
            : focusMax > 0 ? (val / focusMax) * 100 : 0
          const color      = MODEL_COLORS[key] ?? '#94a3b8'
          const isBestHere = key === bestByFocus
          const isBestAUC  = key === bestAUC

          return (
            <div key={key} style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                            alignItems: 'center', marginBottom: 5 }}>
                <span style={{ fontWeight: 600, fontSize: 14, color: color }}>
                  {MODEL_LABELS[key] ?? key}
                  {isBestAUC && (
                    <span style={{ marginLeft: 7, fontSize: 11, color: '#d97706', fontWeight: 700 }}>
                      ★ PRODUCTION
                    </span>
                  )}
                </span>
                <span style={{ fontWeight: 700, fontSize: 14,
                               color: isBestHere ? color : '#374151' }}>
                  {val.toFixed(4)}
                </span>
              </div>
              <div style={{ height: 10, background: '#f1f5f9', borderRadius: 6, overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width:  `${pct}%`,
                  background: color,
                  borderRadius: 6,
                  transition: 'width 0.4s ease',
                  opacity: isBestHere ? 1 : 0.6,
                }} />
              </div>
            </div>
          )
        })}
      </div>

      {/* Full table */}
      <div className="section-heading">Full Metrics Table</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse',
                        fontSize: 13, textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '8px 12px', fontWeight: 700 }}>Model</th>
              {Object.values(METRIC_LABELS).map(l => (
                <th key={l} style={{ padding: '8px 12px', fontWeight: 700, whiteSpace: 'nowrap' }}>
                  {l}
                </th>
              ))}
              <th style={{ padding: '8px 12px', fontWeight: 700 }}>Threshold</th>
            </tr>
          </thead>
          <tbody>
            {modelKeys.map((key, i) => {
              const isBest = key === bestAUC
              return (
                <tr key={key} style={{
                  background: isBest ? '#eff6ff' : i % 2 === 0 ? '#fff' : '#f8fafc',
                  borderBottom: '1px solid #e2e8f0',
                }}>
                  <td style={{ padding: '9px 12px', fontWeight: 700,
                               color: MODEL_COLORS[key] ?? '#374151' }}>
                    {MODEL_LABELS[key] ?? key}
                    {isBest && <span style={{ marginLeft: 6, color: '#d97706' }}>★</span>}
                  </td>
                  {Object.keys(METRIC_LABELS).map(metric => (
                    <td key={metric} style={{ padding: '9px 12px' }}>
                      {metrics[key]?.[metric]?.toFixed(4) ?? '—'}
                    </td>
                  ))}
                  <td style={{ padding: '9px 12px' }}>
                    {metrics[key]?.optimal_threshold?.toFixed(2) ?? '0.50'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-muted" style={{ marginTop: 12, fontSize: 12 }}>
        AUC-ROC: higher is better. Brier Score: lower is better.
        Threshold is optimised on the test set to maximise F1.
      </p>
    </div>
  )
}
