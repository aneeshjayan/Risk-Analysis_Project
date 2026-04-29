import { useState } from 'react'
import LoanForm        from './components/LoanForm.jsx'
import ScoreGauge      from './components/ScoreGauge.jsx'
import ShapChart       from './components/ShapChart.jsx'
import Copilot         from './components/Copilot.jsx'
import ModelComparison from './components/ModelComparison.jsx'
import MacroWidget     from './components/MacroWidget.jsx'

export default function App() {
  const [result,     setResult]     = useState(null)
  const [formInputs, setFormInputs] = useState(null)
  const [copilotOpen, setCopilotOpen] = useState(false)
  const [showMetrics, setShowMetrics] = useState(false)

  function handlePrediction(inputs, response) {
    setFormInputs(inputs)
    setResult(response)
    setShowMetrics(false)
    // Auto-open copilot after a short delay so user sees the result first
    setTimeout(() => setCopilotOpen(true), 900)
    setTimeout(() => {
      document.getElementById('results-anchor')?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }

  function handleReset() {
    setResult(null)
    setFormInputs(null)
    setCopilotOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="app-header">
        {/* Logo */}
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
          <rect width="28" height="28" rx="8" fill="rgba(255,255,255,0.15)" />
          <path d="M14 4l9 5v5c0 5-3.5 8.5-9 10-5.5-1.5-9-5-9-10V9l9-5z"
                fill="rgba(255,255,255,0.85)" />
          <path d="M10.5 14l2.5 2.5 4.5-4.5" stroke="#2563eb"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>

        <h1>LoanLens</h1>
        <span className="header-badge">AI Copilot</span>

        <div className="header-right">
          <button
            className={`header-btn ${showMetrics ? 'active' : ''}`}
            onClick={() => setShowMetrics(v => !v)}
          >
            Model Metrics
          </button>
        </div>
      </header>

      <div className="app-wrapper">

        {/* ── Model Metrics (collapsible) ────────────────────────────────── */}
        {showMetrics && (
          <div className="card mt-6 metrics-panel">
            <ModelComparison />
          </div>
        )}

        {/* ── Hero ───────────────────────────────────────────────────────── */}
        <div className="page-hero">
          <h2>Loan Approval Predictor</h2>
          <p>
            Enter your details for an instant AI-powered decision powered by
            XGBoost, LightGBM &amp; Random Forest — with full SHAP explanation.
          </p>
        </div>

        {/* ── Economic Climate (always visible — FRED Dataset 2) ─────────── */}
        <MacroWidget />

        {/* ── Form ───────────────────────────────────────────────────────── */}
        <div className="card mt-5">
          <LoanForm onResult={handlePrediction} />
        </div>

        {/* ── Results ────────────────────────────────────────────────────── */}
        {result && (
          <div id="results-anchor" className="mt-6">
            <div className="results-grid">
              {/* Left — Score Gauge */}
              <ScoreGauge result={result} inputs={formInputs} />

              {/* Right — SHAP Chart */}
              <ShapChart
                shapValues={result.shap_values}
                featureLabels={result.feature_labels}
                decision={result.decision}
              />
            </div>

            {/* Economic context below result — reminds user of macro environment */}
            <div className="mt-5">
              <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase',
                            letterSpacing: '0.6px', color: '#64748b', marginBottom: 8 }}>
                Economic Context at Time of Application
              </div>
              <MacroWidget />
            </div>

            <div className="mt-5 flex-end">
              <button className="btn btn-ghost" onClick={handleReset}>
                ↩ Start New Application
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── Floating Copilot ───────────────────────────────────────────────── */}
      {result && (
        <Copilot
          result={result}
          inputs={formInputs}
          open={copilotOpen}
          onToggle={() => setCopilotOpen(v => !v)}
        />
      )}
    </>
  )
}
