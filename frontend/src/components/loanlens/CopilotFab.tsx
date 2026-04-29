import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Send, Minimize2, MessageSquare, Bot, AlertCircle } from "lucide-react";
import { chat, type ChatContext, type ChatMessage } from "@/lib/api";

const SUGGESTIONS = [
  "Why was I rejected?",
  "How do I improve?",
  "What affects my score most?",
  "Explain DTI",
];

function openingMessage(ctx: ChatContext) {
  if (!ctx.decision) return "Hi! I'm your LoanLens copilot. Ask me anything about your application.";
  const top = ctx.shap_values?.[0];
  const positive = ctx.shap_values?.find((s) => s.shap < 0);
  const pct = Math.round((ctx.probability ?? 0) * 100);
  return (
    `Your application was **${ctx.decision}** with a **${pct}%** approval probability (risk tier: ${ctx.risk_tier}). ` +
    `The strongest factor was **${top?.feature}** (${top?.value}), ${top && top.shap > 0 ? "increasing" : "reducing"} your risk score. ` +
    (positive ? `On the positive side, **${positive.feature}** helped your case. ` : "") +
    `Ask me anything below.`
  );
}

export function CopilotFab({ context }: { context: ChatContext }) {
  const [visible, setVisible] = useState(false);
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unread, setUnread] = useState(1);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Show FAB 900ms after we get a decision
  useEffect(() => {
    if (context.decision) {
      const t = setTimeout(() => {
        setVisible(true);
        setMessages([{ role: "assistant", content: openingMessage(context) }]);
      }, 900);
      return () => clearTimeout(t);
    }
  }, [context.decision, context.probability]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, typing]);

  if (!visible) return null;

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || typing) return;
    setError(null);
    const next = [...messages, { role: "user" as const, content: trimmed }];
    setMessages(next);
    setInput("");
    setTyping(true);
    try {
      const res = await chat(next, context);
      setMessages((m) => [...m, { role: "assistant", content: res.reply }]);
      if (!open) setUnread((u) => u + 1);
    } catch {
      setError("Couldn't reach the copilot. Try again.");
    } finally {
      setTyping(false);
    }
  };

  return (
    <>
      <motion.button
        initial={{ scale: 0, y: 30 }}
        animate={{ scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 18 }}
        onClick={() => { setOpen(true); setUnread(0); }}
        className="fixed bottom-5 right-5 z-50 inline-flex items-center gap-2 rounded-full ll-gradient-cta text-white pl-3 pr-4 py-3 shadow-xl shadow-blue-700/40 hover:shadow-blue-700/60 transition"
      >
        <span className="relative flex h-6 w-6 items-center justify-center rounded-full bg-white/15">
          <MessageSquare className="h-3.5 w-3.5" />
        </span>
        <span className="text-sm font-semibold">Ask AI</span>
        {unread > 0 && (
          <span className="ml-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-red-500 px-1.5 text-[10px] font-bold">{unread}</span>
        )}
      </motion.button>

      <AnimatePresence>
        {open && (
          <motion.div
            key="panel"
            initial={{ opacity: 0, y: 40, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 30, scale: 0.96 }}
            transition={{ type: "spring", stiffness: 220, damping: 22 }}
            className="fixed bottom-5 right-5 z-50 w-[calc(100vw-2.5rem)] sm:w-[400px] h-[560px] max-h-[80vh] flex flex-col rounded-2xl bg-white border border-slate-200 shadow-2xl overflow-hidden"
          >
            <div className="ll-gradient-header text-white px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/15">
                  <Sparkles className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-sm font-bold leading-tight">LoanLens Copilot</div>
                  <div className="text-[11px] text-white/70 leading-tight">Powered by AI · explains every decision</div>
                </div>
              </div>
              <button onClick={() => setOpen(false)} className="rounded-md p-1 hover:bg-white/10">
                <Minimize2 className="h-4 w-4" />
              </button>
            </div>

            <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3 bg-slate-50">
              {messages.map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={m.role === "user" ? "flex justify-end" : "flex justify-start gap-2"}
                >
                  {m.role !== "user" && (
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white">
                      <Bot className="h-3.5 w-3.5" />
                    </div>
                  )}
                  <div className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${m.role === "user" ? "bg-blue-600 text-white rounded-br-sm" : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm shadow-sm"}`}
                       dangerouslySetInnerHTML={{ __html: m.content.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br/>") }}
                  />
                </motion.div>
              ))}

              {messages.length === 1 && !typing && (
                <div className="flex flex-wrap gap-2 pt-1">
                  {SUGGESTIONS.map((s, i) => (
                    <motion.button
                      key={s}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 + i * 0.06 }}
                      onClick={() => send(s)}
                      className="rounded-full bg-white border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 hover:border-blue-400 hover:text-blue-700 transition"
                    >
                      {s}
                    </motion.button>
                  ))}
                </div>
              )}

              {typing && (
                <div className="flex justify-start gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-white">
                    <Bot className="h-3.5 w-3.5" />
                  </div>
                  <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-sm px-3 py-2.5 shadow-sm flex items-center gap-1">
                    {[0,1,2].map((i) => (
                      <motion.span key={i}
                        animate={{ opacity: [0.3, 1, 0.3], y: [0, -2, 0] }}
                        transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
                        className="h-1.5 w-1.5 rounded-full bg-slate-400"
                      />
                    ))}
                  </div>
                </div>
              )}

              {error && (
                <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 p-2 text-xs text-red-700">
                  <AlertCircle className="h-3.5 w-3.5" /> {error}
                </div>
              )}
            </div>

            <form
              onSubmit={(e) => { e.preventDefault(); send(input); }}
              className="border-t border-slate-200 bg-white p-3 flex items-center gap-2"
            >
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about your decision…"
                className="flex-1 h-10 rounded-lg border border-slate-200 px-3 text-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 outline-none"
              />
              <button
                type="submit"
                disabled={typing || !input.trim()}
                className="h-10 w-10 rounded-lg ll-gradient-cta text-white inline-flex items-center justify-center disabled:opacity-50"
              >
                <Send className="h-4 w-4" />
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
