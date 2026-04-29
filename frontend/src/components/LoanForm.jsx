import { useState } from 'react'

const DEFAULTS = {
  loan_amnt:        '',
  term:             '36',
  annual_inc:       '',
  emp_length:       '5 years',
  home_ownership:   'RENT',
  purpose:          'debt_consolidation',
  application_type: 'Individual',
  credit_score:     '',
  dti:              '',
  credit_age_years: '',
  inq_last_6mths:   '0',
  delinq_2yrs:      '0',
}

const EMP_OPTIONS = [
  '< 1 year', '1 year', '2 years', '3 years', '4 years',
  '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years',
]

const PURPOSE_OPTIONS = [
  { value: 'debt_consolidation', label: 'Debt Consolidation'  },
  { value: 'credit_card',        label: 'Credit Card'         },
  { value: 'home_improvement',   label: 'Home Improvement'    },
  { value: 'car',                label: 'Car Purchase'        },
  { value: 'house',              label: 'House Purchase'      },
  { value: 'major_purchase',     label: 'Major Purchase'      },
  { value: 'small_business',     label: 'Small Business'      },
  { value: 'medical',            label: 'Medical Expenses'    },
  { value: 'moving',             label: 'Moving'              },
  { value: 'vacation',           label: 'Vacation'            },
  { value: 'renewable_energy',   label: 'Renewable Energy'    },
  { value: 'educational',        label: 'Educational'         },
  { value: 'other',              label: 'Other'               },
]

export default function LoanForm({ onResult }) {
  const [form,    setForm]    = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  function handleChange(e) {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)

    const payload = {
      loan_amnt:        parseFloat(form.loan_amnt),
      term:             parseInt(form.term),
      annual_inc:       parseFloat(form.annual_inc),
      emp_length:       form.emp_length,
      home_ownership:   form.home_ownership,
      purpose:          form.purpose,
      application_type: form.application_type,
      credit_score:     parseFloat(form.credit_score),
      dti:              parseFloat(form.dti),
      credit_age_years: parseFloat(form.credit_age_years),
      inq_last_6mths:   parseFloat(form.inq_last_6mths),
      delinq_2yrs:      parseFloat(form.delinq_2yrs),
    }

    // basic validation
    for (const [k, v] of Object.entries(payload)) {
      if (v === null || v === undefined || (typeof v === 'number' && isNaN(v))) {
        setError(`Please fill in all fields. Missing: ${k}`)
        setLoading(false)
        return
      }
    }

    try {
      const res = await fetch('/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      })
      if (!res.ok) {
        let msg = `Server error ${res.status}`
        try {
          const detail = await res.json()
          msg = detail.detail || JSON.stringify(detail) || msg
        } catch {
          msg = await res.text().catch(() => msg)
        }
        throw new Error(msg)
      }
      const data = await res.json()
      onResult(payload, data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      {/* ── Section 1: Loan Details ─────────────────────────────────────── */}
      <div className="section-heading">Loan Details</div>
      <div className="form-grid">

        <div className="field">
          <label>Loan Amount ($)</label>
          <input
            type="number" name="loan_amnt" min="500" max="40000"
            placeholder="e.g. 15000"
            value={form.loan_amnt} onChange={handleChange} required
          />
          <span className="hint">Between $500 and $40,000</span>
        </div>

        <div className="field">
          <label>Loan Term</label>
          <select name="term" value={form.term} onChange={handleChange}>
            <option value="36">36 months (3 years)</option>
            <option value="60">60 months (5 years)</option>
          </select>
        </div>

        <div className="field">
          <label>Loan Purpose</label>
          <select name="purpose" value={form.purpose} onChange={handleChange}>
            {PURPOSE_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label>Application Type</label>
          <select name="application_type" value={form.application_type} onChange={handleChange}>
            <option value="Individual">Individual</option>
            <option value="Joint App">Joint Application</option>
          </select>
        </div>
      </div>

      <hr className="divider" />

      {/* ── Section 2: Financial Profile ──────────────────────────────── */}
      <div className="section-heading">Financial Profile</div>
      <div className="form-grid">

        <div className="field">
          <label>Annual Income ($)</label>
          <input
            type="number" name="annual_inc" min="1000"
            placeholder="e.g. 65000"
            value={form.annual_inc} onChange={handleChange} required
          />
        </div>

        <div className="field">
          <label>Debt-to-Income Ratio (%)</label>
          <input
            type="number" name="dti" min="0" max="100" step="0.1"
            placeholder="e.g. 18.5"
            value={form.dti} onChange={handleChange} required
          />
          <span className="hint">Total monthly debt / monthly income × 100</span>
        </div>

        <div className="field">
          <label>Home Ownership</label>
          <select name="home_ownership" value={form.home_ownership} onChange={handleChange}>
            <option value="RENT">Renting</option>
            <option value="MORTGAGE">Mortgage</option>
            <option value="OWN">Own Outright</option>
            <option value="NONE">None</option>
            <option value="OTHER">Other</option>
          </select>
        </div>

        <div className="field">
          <label>Employment Length</label>
          <select name="emp_length" value={form.emp_length} onChange={handleChange}>
            {EMP_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
        </div>
      </div>

      <hr className="divider" />

      {/* ── Section 3: Credit History ────────────────────────────────── */}
      <div className="section-heading">Credit History</div>
      <div className="form-grid">

        <div className="field">
          <label>Credit Score (FICO)</label>
          <input
            type="number" name="credit_score" min="300" max="850"
            placeholder="e.g. 700"
            value={form.credit_score} onChange={handleChange} required
          />
          <span className="hint">300 (poor) – 850 (excellent)</span>
        </div>

        <div className="field">
          <label>Years of Credit History</label>
          <input
            type="number" name="credit_age_years" min="0" max="60" step="0.5"
            placeholder="e.g. 10"
            value={form.credit_age_years} onChange={handleChange} required
          />
          <span className="hint">How long since your first credit account</span>
        </div>

        <div className="field">
          <label>Credit Inquiries (Last 6 Months)</label>
          <input
            type="number" name="inq_last_6mths" min="0" max="30"
            placeholder="e.g. 1"
            value={form.inq_last_6mths} onChange={handleChange}
          />
          <span className="hint">Hard inquiries from new credit applications</span>
        </div>

        <div className="field">
          <label>Late Payments (Last 2 Years)</label>
          <input
            type="number" name="delinq_2yrs" min="0" max="30"
            placeholder="e.g. 0"
            value={form.delinq_2yrs} onChange={handleChange}
          />
          <span className="hint">Number of 30+ day delinquencies</span>
        </div>
      </div>

      {/* ── Error ──────────────────────────────────────────────────────── */}
      {error && (
        <div style={{
          marginTop: 16, padding: '12px 16px',
          background: '#fee2e2', color: '#dc2626',
          borderRadius: 8, fontSize: 14,
        }}>
          {error}
        </div>
      )}

      {/* ── Submit ─────────────────────────────────────────────────────── */}
      <div className="mt-6 flex-end">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? <><span className="spinner" /> Analyzing…</> : 'Predict Loan Outcome'}
        </button>
      </div>
    </form>
  )
}
