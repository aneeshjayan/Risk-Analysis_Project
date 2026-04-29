import { motion } from "framer-motion";
import { CheckCircle2, XCircle, Star } from "lucide-react";
import type { PredictResponse, LoanInput } from "@/lib/api";
import { fmtCurrency, fmtPct, tierColor, purposeLabel } from "@/lib/loan-utils";

export function ScoreGauge({ result, input }: { result: PredictResponse; input: LoanInput }) {
  const approved = result.decision === "APPROVED";
  const pct = Math.round(result.probability * 100);
  const radius = 90;
  const stroke = 14;
  const c = 2 * Math.PI * radius;
  const offset = c * (1 - result.probability);
  const color = approved ? "#10b981" : "#dc2626";
  const ficoMid = Math.round((input.fico_range_low + input.fico_range_high) / 2);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white ll-shadow-soft p-6 sm:p-8">
      <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Decision</div>
      <h3 className="text-xl font-bold text-slate-900 mb-6">Approval Probability</h3>

      <div className="relative flex items-center justify-center">
        <svg width="220" height="220" viewBox="0 0 220 220" className="-rotate-90">
          <circle cx="110" cy="110" r={radius} strokeWidth={stroke} stroke="#f1f5f9" fill="none" />
          <motion.circle
            cx="110" cy="110" r={radius} strokeWidth={stroke} stroke={color} fill="none" strokeLinecap="round"
            strokeDasharray={c}
            initial={{ strokeDashoffset: c }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.2, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.6 }}
            className="text-5xl font-bold tabular-nums" style={{ color }}
          >
            {pct}%
          </motion.div>
          <div className="text-xs text-slate-500 mt-1">approval likelihood</div>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap justify-center gap-2">
        <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-bold ${approved ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-red-50 text-red-700 border border-red-200"}`}>
          {approved ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
          {approved ? "APPROVED" : "REJECTED"}
        </span>
        <span className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold border ${tierColor(result.risk_tier)}`}>
          {result.risk_tier} Risk
        </span>
      </div>

      <div className="mt-5 grid grid-cols-3 gap-2">
        {(Object.entries(result.model_probs) as [keyof typeof result.model_probs, number][]).map(([m, p]) => {
          const winner = m === result.best_model;
          return (
            <div key={m} className={`rounded-lg border px-2 py-2 text-center ${winner ? "border-amber-300 bg-amber-50" : "border-slate-200 bg-slate-50"}`}>
              <div className="text-[10px] uppercase font-semibold tracking-wide text-slate-500 flex items-center justify-center gap-1">
                {winner && <Star className="h-3 w-3 fill-amber-500 text-amber-500" />}
                {m}
              </div>
              <div className="text-sm font-bold tabular-nums text-slate-900 mt-0.5">{fmtPct(p, 0)}</div>
            </div>
          );
        })}
      </div>

      <div className="mt-6 border-t border-slate-100 pt-5">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">Profile snapshot</div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          {[
            ["Loan amount", fmtCurrency(input.loan_amnt)],
            ["Annual income", fmtCurrency(input.annual_inc)],
            ["FICO (mid)", String(ficoMid)],
            ["DTI", `${input.dti}%`],
            ["Term", `${input.term} mo`],
            ["Purpose", purposeLabel(input.purpose)],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg bg-slate-50 px-3 py-2">
              <div className="text-[10px] uppercase tracking-wide text-slate-500">{k}</div>
              <div className="font-semibold text-slate-900 truncate">{v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
