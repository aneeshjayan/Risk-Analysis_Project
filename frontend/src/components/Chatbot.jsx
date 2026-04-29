import { useState, useRef, useEffect } from 'react'

const SUGGESTIONS_ACCEPTED = [
  'What were my strongest factors?',
  'How can I get a lower interest rate?',
  'What does my debt-to-income ratio mean?',
  'Should I apply for a larger amount?',
]

const SUGGESTIONS_REJECTED = [
  'Why was my loan rejected?',
  'How can I improve my chances?',
  'How do I improve my credit score?',
  'What DTI ratio should I aim for?',
  'How long until I should reapply?',
]

const INITIAL_MESSAGE = (decision) => ({
  role:    'assistant',
  content: decision === 'Accepted'
    ? "Great news — your application looks strong! I'm your AI loan advisor. Feel free to ask me anything about your result, credit profile, or how to get the best loan terms."
    : "I'm here to help you understand this outcome and improve your chances. Ask me anything about the decision, your credit profile, or what steps you can take to get approved.",
})

export default function Chatbot({ result, inputs }) {
  const [messages, setMessages] = useState([INITIAL_MESSAGE(result.decision)])
  const [input,    setInput]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const suggestions = result.decision === 'Accepted'
    ? SUGGESTIONS_ACCEPTED
    : SUGGESTIONS_REJECTED

  async function sendMessage(text) {
    const userMsg = text.trim()
    if (!userMsg || loading) return

    setInput('')
    setError(null)

    const updatedMessages = [...messages, { role: 'user', content: userMsg }]
    setMessages(updatedMessages)
    setLoading(true)

    const context = {
      decision:            result.decision,
      probability:         result.probability,
      model_probabilities: result.model_probabilities,
      inputs:              inputs,
      shap_values:         result.shap_values,
    }

    try {
      const res = await fetch('/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updatedMessages,
          context,
        }),
      })

      if (!res.ok) {
        let msg = `Server error ${res.status}`
        try {
          const detail = await res.json()
          msg = detail.detail || msg
        } catch {
          msg = await res.text().catch(() => msg)
        }
        throw new Error(msg)
      }

      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }])
    } catch (err) {
      setError(err.message)
      setMessages(prev => prev.slice(0, -1))   // remove optimistic user msg
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
    <div>
      {/* Suggestion chips (only before first user message) */}
      {messages.length === 1 && (
        <div className="suggestions">
          {suggestions.map(s => (
            <button
              key={s}
              className="suggestion-chip"
              onClick={() => sendMessage(s)}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="chat-window">
        {/* Messages */}
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`chat-bubble ${msg.role}`}
            >
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
              Error: {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="chat-input-row">
          <input
            type="text"
            placeholder="Ask anything about your loan decision…"
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
    </div>
  )
}
