/**
 * MacroWidget — shows current FRED macroeconomic conditions.
 * Fetched from /macro endpoint (live Federal Reserve data).
 * Displayed alongside the prediction result to give economic context.
 */
import { useState, useEffect } from 'react'

const STATUS_COLORS = {
  good: { bg: '#dcfce7', text: '#15803d', dot: '#22c55e' },
  warn: { bg: '#fef9c3', text: '#a16207', dot: '#f59e0b' },
  bad:  { bg: '#fee2e2', text: '#b91c1c', dot: '#ef4444' },
}

const TREND_ICON = {
  rising:  '↑',
  falling: '↓',
  stable:  '→',
}

const CLIMATE_COLORS = {
  green:  { bg: 'linear-gradient(135deg,#dcfce7,#bbf7d0)', border: '#86efac', text: '#15803d' },
  orange: { bg: 'linear-gradient(135deg,#fef9c3,#fde68a)', border: '#fcd34d', text: '#92400e' },
  red:    { bg: 'linear-gradient(135deg,#fee2e2,#fecaca)', border: '#fca5a5', text: '#991b1b' },
}

export default function MacroWidget() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    fetch('/macro')
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return (
    <div style={{ padding: '14px 20px', color: '#64748b', fontSize: 13 }}>
      Loading economic conditions…
    </div>
  )
  if (error) return null   // fail silently — FRED is optional

  const { indicators, climate, impact_on_loan } = data
  const cc = CLIMATE_COLORS[climate.color] || CLIMATE_COLORS.green

  return (
    <div style={{
      background: cc.bg,
      border: `1px solid ${cc.border}`,
      borderRadius: 14,
      overflow: 'hidden',
      marginTop: 20,
    }}>

      {/* ── Climate header ─────────────────────────────────────────────── */}
      <div
        style={{
          background: cc.bg,
          borderBottom: `1px solid ${cc.border}`,
          padding: '14px 20px',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          cursor: 'pointer',
        }}
        onClick={() => setExpanded(v => !v)}
      >
        <div style={{ fontSize: 22 }}>
          {climate.color === 'green' ? '🌤' : climate.color === 'orange' ? '⚠️' : '🔴'}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: cc.text }}>
            Economic Climate: {climate.label}
          </div>
          <div style={{ fontSize: 12.5, color: cc.text, opacity: 0.85, marginTop: 2 }}>
            {climate.message}
          </div>
        </div>
        <div style={{ fontSize: 12, color: cc.text, opacity: 0.7, fontWeight: 600 }}>
          {expanded ? 'Hide ▲' : 'Details ▼'}
        </div>
      </div>

      {/* ── Impact bullets — always visible ────────────────────────────── */}
      <div style={{ padding: '10px 20px', background: 'rgba(255,255,255,0.5)' }}>
        <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
                      letterSpacing: '0.6px', color: '#64748b', marginBottom: 6 }}>
          How this affects your application
        </div>
        {impact_on_loan.map((line, i) => (
          <div key={i} style={{ fontSize: 13, marginBottom: 4, lineHeight: 1.5 }}>
            {line}
          </div>
        ))}
      </div>

      {/* ── Expanded indicator grid ─────────────────────────────────────── */}
      {expanded && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
          gap: 12,
          padding: '14px 20px 18px',
          background: 'rgba(255,255,255,0.6)',
          borderTop: `1px solid ${cc.border}`,
        }}>
          {Object.entries(indicators).map(([key, ind]) => {
            const sc = STATUS_COLORS[ind.status] || STATUS_COLORS.good
            const trendIcon = TREND_ICON[ind.trend] || '→'
            const trendColor = ind.trend === 'rising' ? '#ef4444'
                             : ind.trend === 'falling' ? '#22c55e'
                             : '#94a3b8'
            return (
              <div key={key} style={{
                background: sc.bg,
                border: `1px solid ${sc.dot}33`,
                borderRadius: 10,
                padding: '12px 14px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center',
                              justifyContent: 'space-between', marginBottom: 4 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
                                letterSpacing: '0.5px', color: sc.text }}>
                    {ind.label}
                  </div>
                  <div style={{ width: 8, height: 8, borderRadius: '50%',
                                background: sc.dot, flexShrink: 0 }} />
                </div>

                <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                  <span style={{ fontSize: 22, fontWeight: 800, color: sc.text, lineHeight: 1 }}>
                    {key === 'recession_flag'
                      ? (ind.value ? 'Yes' : 'No')
                      : key === 'real_disposable_inc'
                        ? `$${Number(ind.value).toLocaleString()}`
                        : ind.value}
                  </span>
                  {ind.unit && key !== 'recession_flag' && key !== 'real_disposable_inc' && (
                    <span style={{ fontSize: 13, color: sc.text, opacity: 0.7 }}>{ind.unit}</span>
                  )}
                  {ind.trend !== 'stable' && (
                    <span style={{ fontSize: 14, color: trendColor, fontWeight: 700 }}>
                      {trendIcon}
                      {Math.abs(ind.change) > 0.01 && (
                        <span style={{ fontSize: 11 }}> {ind.change > 0 ? '+' : ''}{ind.change}</span>
                      )}
                    </span>
                  )}
                </div>

                <div style={{ fontSize: 11, color: sc.text, opacity: 0.75,
                              marginTop: 5, lineHeight: 1.4 }}>
                  {ind.note}
                </div>
              </div>
            )
          })}

          <div style={{ gridColumn: '1/-1', fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
            📡 Source: Federal Reserve Economic Data (FRED) — St. Louis Fed
          </div>
        </div>
      )}
    </div>
  )
}
