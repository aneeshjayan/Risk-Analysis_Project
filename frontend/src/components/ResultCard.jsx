export default function ResultCard({ result, inputs }) {
  const accepted = result.decision === 'Accepted'
  const prob     = result.probability          // probability of DEFAULT (1 = bad)
  const displayProb = accepted
    ? ((1 - prob) * 100).toFixed(1)            // approval confidence
    : (prob * 100).toFixed(1)                  // risk level

  return (
    <div>
      {/* ── Decision banner ────────────────────────────────────────────── */}
      <div className={`result-card ${accepted ? 'result-accepted' : 'result-rejected'}`}>
        <div className="result-icon">{accepted ? '✅' : '❌'}</div>
        <div className="result-label">
          {accepted ? 'Loan Likely Approved' : 'Loan Likely Rejected'}
        </div>
        <div className="result-prob">
          {accepted
            ? `Approval confidence: ${displayProb}%`
            : `Default risk: ${displayProb}%`}
        </div>

        {/* Probability bar */}
        <div className="prob-bar-wrap">
          <div
            className={`prob-bar-fill ${accepted ? 'accepted-fill' : 'rejected-fill'}`}
            style={{ width: `${displayProb}%` }}
          />
        </div>
      </div>

      {/* ── Application summary ────────────────────────────────────────── */}
      <div className="mt-6">
        <div className="section-heading">Application Summary</div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
          gap: '12px',
        }}>
          {[
            ['Loan Amount',         `$${Number(inputs.loan_amnt).toLocaleString()}`],
            ['Loan Term',           `${inputs.term} months`],
            ['Annual Income',       `$${Number(inputs.annual_inc).toLocaleString()}`],
            ['Employment Length',   inputs.emp_length],
            ['Home Ownership',      inputs.home_ownership],
            ['Loan Purpose',        inputs.purpose.replace(/_/g, ' ')],
            ['Application Type',    inputs.application_type],
            ['Credit Score',        inputs.credit_score],
            ['DTI Ratio',           `${inputs.dti}%`],
            ['Credit History',      `${inputs.credit_age_years} yrs`],
            ['Recent Inquiries',    inputs.inq_last_6mths],
            ['Late Payments (2yr)', inputs.delinq_2yrs],
          ].map(([label, value]) => (
            <SummaryItem key={label} label={label} value={value} />
          ))}
        </div>
      </div>

      {/* ── Model Agreement ────────────────────────────────────────────── */}
      {result.model_probabilities && (
        <div className="mt-6">
          <div className="section-heading">Model Agreement (Ensemble of 3)</div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
            gap: '12px',
          }}>
            {Object.entries(result.model_probabilities).map(([model, prob]) => {
              const isRisk = prob >= 0.5
              return (
                <div key={model} style={{
                  background: '#f8fafc',
                  border: `1.5px solid ${model === 'ensemble' ? '#2563eb' : '#e2e8f0'}`,
                  borderRadius: 8,
                  padding: '10px 14px',
                }}>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600,
                                textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
                    {model.replace(/_/g, ' ')}
                    {model === 'ensemble' && ' ★'}
                  </div>
                  <div style={{ fontWeight: 700, fontSize: 16,
                                color: isRisk ? '#dc2626' : '#16a34a' }}>
                    {(prob * 100).toFixed(1)}% risk
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                    {isRisk ? 'leans Reject' : 'leans Accept'}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── Hint ───────────────────────────────────────────────────────── */}
      <p className="mt-4 text-muted">
        {accepted
          ? 'Switch to the "Why?" tab to see which factors drove this decision.'
          : 'Switch to "Why? (SHAP)" to understand the key risk factors, or ask the AI Advisor for improvement tips.'}
      </p>
    </div>
  )
}

function SummaryItem({ label, value }) {
  return (
    <div style={{
      background: '#f8fafc',
      border: '1px solid #e2e8f0',
      borderRadius: 8,
      padding: '10px 14px',
    }}>
      <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600,
                    textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 2 }}>
        {label}
      </div>
      <div style={{ fontWeight: 600, fontSize: 14, textTransform: 'capitalize' }}>
        {String(value)}
      </div>
    </div>
  )
}
