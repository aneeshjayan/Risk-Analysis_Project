/**
 * ModelComparison — fetches /models/metrics and renders a performance table
 * and a visual bar comparison for all trained models (XGBoost, LightGBM,
 * Random Forest, Ensemble).
 */
import { useState, useEffect } from 'react'

const METRIC_LABELS = {
  auc_roc:   'AUC-ROC',
  f1:        'F1 Score',
  precision: 'Precision',
  recall:    'Recall',
  brier:     'Brier Score',
}

const MODEL_COLORS = {
  xgboost:       '#2563eb',
  lightgbm:      '#7c3aed',
  random_forest: '#059669',
  ensemble:      '#d97706',
}

const MODEL_LABELS = {
  xgboost:       'XGBoost',
  lightgbm:      'LightGBM',
  random_forest: 'Random Forest',
  ensemble:      'Soft Ensemble',
}

export default function ModelComparison() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [focus,   setFocus]   = useState('auc_roc')

  useEffect(() => {
    fetch('/models/metrics')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(data => { setMetrics(data); setLoading(false) })
      .catch(e   => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <p style={{ padding: 24, color: '#64748b' }}>Loading model metrics…</p>
  if (error)   return (
    <div style={{ padding: 24, color: '#dc2626' }}>
      <strong>Could not load metrics.</strong> Run <code>python train_model.py</code> first.
      <br /><small>{error}</small>
    </div>
  )

  const modelKeys = Object.keys(metrics)

  // Best model by AUC-ROC
  const bestModel = modelKeys.reduce((a, b) =>
    (metrics[a]?.auc_roc ?? 0) > (metrics[b]?.auc_roc ?? 0) ? a : b
  )

  const focusMax = Math.max(...modelKeys.map(k => metrics[k]?.[focus] ?? 0))

  return (
    <div>
      <div className="section-heading">Model Performance Comparison</div>
      <p className="text-muted" style={{ marginBottom: 20, fontSize: 13 }}>
        All three base models trained on 80% of LendingClub data, evaluated on a held-out 20% test set.
        The Soft Ensemble averages probabilities from all three models.
      </p>

      {/* ── Metric selector ──────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
        {Object.entries(METRIC_LABELS).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFocus(key)}
            style={{
              padding:      '5px 14px',
              borderRadius: 6,
              border:       `1.5px solid ${focus === key ? '#2563eb' : '#dde3eb'}`,
              background:   focus === key ? '#2563eb' : '#fff',
              color:        focus === key ? '#fff'    : '#374151',
              fontWeight:   600,
              fontSize:     13,
              cursor:       'pointer',
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Visual bar comparison for selected metric ─────────────────────── */}
      <div style={{ marginBottom: 28 }}>
        {modelKeys.map(key => {
          const val    = metrics[key]?.[focus] ?? 0
          const pct    = focusMax > 0 ? (val / focusMax) * 100 : 0
          const color  = MODEL_COLORS[key] ?? '#94a3b8'
          const isBest = key === bestModel && focus === 'auc_roc'
          const lowerIsBetter = focus === 'brier'
          const isMetricBest = lowerIsBetter
            ? val === Math.min(...modelKeys.map(k => metrics[k]?.[focus] ?? 1))
            : val === focusMax

          return (
            <div key={key} style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                            alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontWeight: 600, fontSize: 14 }}>
                  {MODEL_LABELS[key] ?? key}
                  {isBest && (
                    <span style={{ marginLeft: 8, fontSize: 11, color: '#d97706',
                                   fontWeight: 700 }}>★ BEST</span>
                  )}
                </span>
                <span style={{ fontWeight: 700, color: isMetricBest ? color : '#374151' }}>
                  {val.toFixed(4)}
                </span>
              </div>
              <div style={{ height: 10, background: '#f1f5f9', borderRadius: 5, overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width:  lowerIsBetter ? `${(1 - val / focusMax) * 100}%` : `${pct}%`,
                  background: color,
                  borderRadius: 5,
                  transition: 'width 0.35s ease',
                }} />
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Full metrics table ────────────────────────────────────────────── */}
      <div className="section-heading">Full Metrics Table</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{
          width: '100%', borderCollapse: 'collapse',
          fontSize: 13, textAlign: 'left',
        }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '8px 12px', fontWeight: 700, color: '#374151' }}>Model</th>
              {Object.values(METRIC_LABELS).map(l => (
                <th key={l} style={{ padding: '8px 12px', fontWeight: 700,
                                     color: '#374151', whiteSpace: 'nowrap' }}>{l}</th>
              ))}
              <th style={{ padding: '8px 12px', fontWeight: 700, color: '#374151' }}>CV AUC</th>
              <th style={{ padding: '8px 12px', fontWeight: 700, color: '#374151' }}>CV F1</th>
            </tr>
          </thead>
          <tbody>
            {modelKeys.map((key, i) => (
              <tr key={key}
                  style={{
                    background: key === bestModel ? '#eff6ff' : i % 2 === 0 ? '#fff' : '#f8fafc',
                    borderBottom: '1px solid #e2e8f0',
                  }}>
                <td style={{ padding: '9px 12px', fontWeight: 600,
                             color: MODEL_COLORS[key] ?? '#374151' }}>
                  {MODEL_LABELS[key] ?? key}
                  {key === bestModel && <span style={{ marginLeft: 6, color: '#d97706' }}>★</span>}
                </td>
                {Object.keys(METRIC_LABELS).map(metric => (
                  <td key={metric} style={{ padding: '9px 12px' }}>
                    {metrics[key]?.[metric]?.toFixed(4) ?? '—'}
                  </td>
                ))}
                <td style={{ padding: '9px 12px' }}>
                  {metrics[key]?.cv_auc_mean != null
                    ? `${metrics[key].cv_auc_mean.toFixed(4)} ± ${metrics[key].cv_auc_std?.toFixed(4)}`
                    : '—'}
                </td>
                <td style={{ padding: '9px 12px' }}>
                  {metrics[key]?.cv_f1_mean != null
                    ? `${metrics[key].cv_f1_mean.toFixed(4)} ± ${metrics[key].cv_f1_std?.toFixed(4)}`
                    : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-muted" style={{ marginTop: 14, fontSize: 12 }}>
        AUC-ROC: higher is better. Brier Score: lower is better.
        CV = 3-fold stratified cross-validation on training set.
      </p>
    </div>
  )
}
