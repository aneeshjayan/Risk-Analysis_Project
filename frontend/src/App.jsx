import { useState } from 'react'
import LoanForm        from './components/LoanForm.jsx'
import ResultCard      from './components/ResultCard.jsx'
import ShapChart       from './components/ShapChart.jsx'
import Chatbot         from './components/Chatbot.jsx'
import ModelComparison from './components/ModelComparison.jsx'

export default function App() {
  const [result,      setResult]      = useState(null)
  const [formInputs,  setFormInputs]  = useState(null)
  const [activeTab,   setActiveTab]   = useState('result')
  const [showModels,  setShowModels]  = useState(false)

  function handlePrediction(inputs, response) {
    setFormInputs(inputs)
    setResult(response)
    setActiveTab('result')
    setShowModels(false)
    setTimeout(() => {
      document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }

  function handleReset() {
    setResult(null)
    setFormInputs(null)
    setShowModels(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="app-header">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <rect width="24" height="24" rx="6" fill="rgba(255,255,255,0.2)" />
          <path d="M12 4l7 4v4c0 4-3 7-7 8-4-1-7-4-7-8V8l7-4z"
                fill="rgba(255,255,255,0.9)" />
          <path d="M9.5 12l2 2 3-3" stroke="#2563eb" strokeWidth="1.8"
                strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <h1>LoanLens</h1>
        <span className="header-badge">AI-Powered</span>

        {/* Model comparison toggle in header */}
        <button
          onClick={() => setShowModels(v => !v)}
          style={{
            marginLeft:   'auto',
            padding:      '5px 14px',
            borderRadius: 6,
            border:       '1.5px solid rgba(255,255,255,0.4)',
            background:   showModels ? 'rgba(255,255,255,0.25)' : 'transparent',
            color:        '#fff',
            fontWeight:   600,
            fontSize:     13,
            cursor:       'pointer',
          }}
        >
          Model Metrics
        </button>
      </header>

      <div className="app-wrapper">

        {/* ── Model Comparison panel (toggleable) ───────────────────────── */}
        {showModels && (
          <div className="card" style={{ marginBottom: 24 }}>
            <ModelComparison />
          </div>
        )}

        {/* ── Hero ───────────────────────────────────────────────────────── */}
        <div className="page-title">
          <h2>Loan Approval Predictor</h2>
          <p>
            Fill in your details below to get an instant AI-powered loan decision
            using an ensemble of <strong>XGBoost</strong>, <strong>LightGBM</strong>,
            and <strong>Random Forest</strong> models with full SHAP explanation.
          </p>
        </div>

        {/* ── Loan Form ──────────────────────────────────────────────────── */}
        <div className="card">
          <LoanForm onResult={handlePrediction} />
        </div>

        {/* ── Results section ────────────────────────────────────────────── */}
        {result && (
          <div id="result-section" className="mt-6">

            <div className="card">
              {/* Tabs */}
              <div className="tabs">
                <button
                  className={`tab-btn ${activeTab === 'result' ? 'active' : ''}`}
                  onClick={() => setActiveTab('result')}
                >
                  Decision
                </button>
                <button
                  className={`tab-btn ${activeTab === 'shap' ? 'active' : ''}`}
                  onClick={() => setActiveTab('shap')}
                >
                  Why? (SHAP)
                </button>
                <button
                  className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
                  onClick={() => setActiveTab('chat')}
                >
                  Ask AI Advisor
                </button>
                <button
                  className={`tab-btn ${activeTab === 'models' ? 'active' : ''}`}
                  onClick={() => setActiveTab('models')}
                >
                  Models
                </button>
              </div>

              {/* Tab: Decision */}
              {activeTab === 'result' && (
                <ResultCard result={result} inputs={formInputs} />
              )}

              {/* Tab: SHAP */}
              {activeTab === 'shap' && (
                <ShapChart
                  shapValues={result.shap_values}
                  featureLabels={result.feature_labels}
                  decision={result.decision}
                />
              )}

              {/* Tab: Chatbot */}
              {activeTab === 'chat' && (
                <Chatbot result={result} inputs={formInputs} />
              )}

              {/* Tab: Model Comparison */}
              {activeTab === 'models' && (
                <ModelComparison />
              )}
            </div>

            {/* Reset button */}
            <div className="mt-4 flex-end">
              <button className="btn btn-ghost" onClick={handleReset}>
                Start New Application
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  )
}
