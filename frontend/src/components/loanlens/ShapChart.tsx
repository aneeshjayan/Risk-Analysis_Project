import { motion } from "framer-motion";
import { ArrowUp, ArrowDown } from "lucide-react";
import type { ShapValue } from "@/lib/api";

export function ShapChart({ shap }: { shap: ShapValue[] }) {
  const max = Math.max(...shap.map((s) => Math.abs(s.shap)), 0.01);
  return (
    <div className="rounded-2xl border border-slate-200 bg-white ll-shadow-soft p-6 sm:p-8">
      <div className="flex items-start justify-between mb-1">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">Explainability</div>
          <h3 className="text-xl font-bold text-slate-900">Top 10 SHAP Factors</h3>
        </div>
        <div className="flex flex-col gap-1 text-[11px]">
          <span className="inline-flex items-center gap-1 text-red-600"><ArrowUp className="h-3 w-3" /> Increases Risk</span>
          <span className="inline-flex items-center gap-1 text-emerald-600"><ArrowDown className="h-3 w-3" /> Reduces Risk</span>
        </div>
      </div>

      <div className="mt-5 space-y-3">
        {shap.map((s, i) => {
          const pct = (Math.abs(s.shap) / max) * 100;
          const positive = s.shap > 0; // increases risk
          return (
            <div key={s.feature}>
              <div className="flex items-baseline justify-between text-xs">
                <div className="font-semibold text-slate-800">{s.feature}</div>
                <div className="tabular-nums text-slate-500">
                  <span className="mr-2">{String(s.value)}</span>
                  <span className={positive ? "text-red-600 font-semibold" : "text-emerald-600 font-semibold"}>
                    {positive ? "+" : ""}{s.shap.toFixed(2)}
                  </span>
                </div>
              </div>
              <div className="mt-1 h-3 rounded-full bg-slate-100 overflow-hidden flex">
                <div className="w-1/2 flex justify-end">
                  {!positive && (
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.5, delay: i * 0.05 }}
                      className="h-full bg-emerald-500 rounded-l-full"
                    />
                  )}
                </div>
                <div className="w-px bg-slate-300" />
                <div className="w-1/2">
                  {positive && (
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.5, delay: i * 0.05 }}
                      className="h-full bg-red-500 rounded-r-full"
                    />
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
