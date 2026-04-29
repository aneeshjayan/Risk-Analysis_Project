import { motion, AnimatePresence } from "framer-motion";
import { Shield, Sparkles, BarChart3, ChevronDown } from "lucide-react";
import { useEffect, useState } from "react";
import { getModelMetrics, type ModelMetrics } from "@/lib/api";
import { Star } from "lucide-react";

export function LoanLensHeader() {
  const [open, setOpen] = useState(false);
  const [metrics, setMetrics] = useState<ModelMetrics[] | null>(null);

  useEffect(() => {
    if (open && !metrics) {
      getModelMetrics().then(setMetrics).catch(() => {});
    }
  }, [open, metrics]);

  return (
    <header className="sticky top-0 z-40 ll-gradient-header text-white border-b border-white/10">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-blue-400 to-blue-700 shadow-lg shadow-blue-900/40">
              <Shield className="h-5 w-5 text-white" strokeWidth={2.5} />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold tracking-tight">LoanLens</span>
              <span className="hidden sm:inline-flex items-center gap-1 rounded-full bg-blue-500/20 border border-blue-400/40 px-2.5 py-0.5 text-[11px] font-semibold text-blue-100">
                <Sparkles className="h-3 w-3" /> AI Copilot
              </span>
            </div>
          </div>

          <button
            onClick={() => setOpen((v) => !v)}
            className="inline-flex items-center gap-2 rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/90 hover:bg-white/10 transition-colors"
          >
            <BarChart3 className="h-4 w-4" />
            Model Metrics
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="overflow-hidden border-t border-white/10 bg-slate-950/40 backdrop-blur"
          >
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
              {!metrics ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {[0,1,2].map((i) => <div key={i} className="h-28 rounded-xl bg-white/5 animate-pulse" />)}
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {metrics.map((m) => (
                    <div key={m.model} className={`relative rounded-xl border p-4 ${m.is_winner ? "border-amber-300/40 bg-amber-300/10" : "border-white/10 bg-white/5"}`}>
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold text-white">{m.model}</div>
                        {m.is_winner && (
                          <span className="inline-flex items-center gap-1 rounded-full bg-amber-300/20 px-2 py-0.5 text-[10px] font-bold text-amber-200 border border-amber-300/40">
                            <Star className="h-3 w-3 fill-amber-300 text-amber-300" /> WINNER
                          </span>
                        )}
                      </div>
                      <div className="mt-3 grid grid-cols-5 gap-2 text-center">
                        {[
                          ["AUC", m.auc_roc],
                          ["F1", m.f1],
                          ["Prec", m.precision],
                          ["Rec", m.recall],
                          ["Brier", m.brier],
                        ].map(([k, v]) => (
                          <div key={k as string}>
                            <div className="text-[10px] uppercase tracking-wider text-white/50">{k}</div>
                            <div className="text-sm font-semibold text-white tabular-nums">{(v as number).toFixed(3)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
