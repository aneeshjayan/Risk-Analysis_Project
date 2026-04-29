/**
 * Copilot — floating AI advisor panel.
 * Opens automatically after prediction, slides up from bottom-right.
 * Explains SHAP factors and answers follow-up questions via OpenAI.
 */
import { useState, useRef, useEffect } from 'react'

const FEATURE_LABELS = {
  loan_amnt:       'Loan Amount',
  annual_inc:      'Annual Income',
  credit_score:    'Credit Score',
  dti:             'Debt-to-Income Ratio',
  emp_length:      'Employment Length',
  credit_age_years:'Credit History',
  inq_last_6mths:  'Credit Inquiries',
  delinq_2yrs:     'Late Payments',
}

function buildInitialGreeting(result) {
  const accepted = result.decision === 'Accepted'
  const pct      = Math.round(result.probability * 100)

  // Find top 2 risk factors from SHAP
  const topFactors = Object.entries(result.shap_values || {})
    .map(([k, v]) => ({ key: k, val: v, abs: Math.abs(v) }))
    .sort((a, b) => b.abs - a.abs)
    .slice(0, 2)
    .map(f => (result.feature_labels?.[f.key] || f.key.replace(/_/g, ' ')))

  if (accepted) {
    return `Great news — your application looks strong with only ${pct}% default risk! 🎉\n\nYour top positive factors were **${topFactors[0]}** and **${topFactors[1]}**.\n\nAsk me anything about your result or how to secure the best loan terms.`
  } else {
    return `Your application shows a ${pct}% default risk, which resulted in a likely rejection.\n\nThe two biggest risk factors were **${topFactors[0]}** and **${topFactors[1]}**.\n\nI can explain each factor and give you a specific action plan to improve your chances. What would you like to know?`
  }
}

const SUGGESTIONS_ACCEPTED = [
  'What are my strongest factors?',
  'How can I get a lower rate?',
  'What does my DTI mean?',
  'Should I borrow more?',
]
const SUGGESTIONS_REJECTED = [
  'How do I improve my credit score?',
  'What DTI should I aim for?',
  'When should I reapply?',
  'Explain the top risk factor',
]

export default function Copilot({ result, inputs, open, onToggle }) {
  const initialMsg = { role: 'assistant', content: buildInitialGreeting(result) }
  const [messages, setMessages] = useState([initialMsg])
  const [input,    setInput]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const bottomRef  = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const suggestions = result.decision === 'Accepted' ? SUGGESTIONS_ACCEPTED : SUGGESTIONS_REJECTED
  const showChips   = messages.length === 1   // only before first user message

  async function sendMessage(text) {
    const userText = text.trim()
    if (!userText || loading) return

    setInput('')
    setError(null)

    const updatedMessages = [...messages, { role: 'user', content: userText }]
    setMessages(updatedMessages)
    setLoading(true)

    try {
      const res = await fetch('/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updatedMessages,
          context: {
            decision:            result.decision,
            probability:         result.probability,
            risk_tier:           result.risk_tier,
            model_probabilities: result.model_probabilities,
            inputs,
            shap_values:         result.shap_values,
          },
        }),
      })

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}))
        throw new Error(detail.detail || `Server error ${res.status}`)
      }

      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }])
    } catch (err) {
      setError(err.message)
      setMessages(prev => prev.slice(0, -1))
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  return (
    <>
      {/* ── Floating action button ──────────────────────────────────────── */}
      <button className="copilot-fab" onClick={onToggle} title="AI Copilot">
        {open ? '✕' : '🤖'}
        {!open && (
          <span className="copilot-badge">
            {result.decision === 'Accepted' ? '✓' : '!'}
          </span>
        )}
      </button>

      {/* ── Slide-up panel ─────────────────────────────────────────────── */}
      {open && (
        <div className="copilot-panel">

          {/* Header */}
          <div className="copilot-header">
            <div className="copilot-avatar">🤖</div>
            <div className="copilot-header-text">
              <h4>LoanLens Copilot</h4>
              <p><span className="online-dot" /> AI Advisor · Powered by GPT-4o-mini</p>
            </div>
            <button className="copilot-close" onClick={onToggle}>✕</button>
          </div>

          {/* Messages */}
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-bubble ${msg.role}`}>
                {msg.content}
              </div>
            ))}

            {loading && (
              <div className="chat-bubble assistant typing">
                Thinking…
              </div>
            )}

            {error && (
              <div className="chat-bubble assistant" style={{ color: '#dc2626', background: '#fee2e2' }}>
                ⚠ {error}
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Suggestion chips */}
          {showChips && (
            <div className="suggestions-row">
              {suggestions.map(s => (
                <button key={s} className="suggestion-chip" onClick={() => sendMessage(s)}>
                  {s}
                </button>
              ))}
            </div>
          )}

          {/* Input bar */}
          <div className="chat-input-row">
            <input
              type="text"
              placeholder="Ask about your result…"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            <button
              className="chat-send-btn"
              onClick={() => sendMessage(input)}
              disabled={loading || !input.trim()}
              title="Send"
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  )
}
