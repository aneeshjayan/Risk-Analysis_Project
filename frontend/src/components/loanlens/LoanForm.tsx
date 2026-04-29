import { useForm, Controller } from "react-hook-form";
import { motion } from "framer-motion";
import { Loader2, ArrowRight } from "lucide-react";
import { PURPOSES, purposeLabel } from "@/lib/loan-utils";
import type { LoanInput } from "@/lib/api";

type FormValues = {
  loan_amnt: number;
  purpose: string;
  annual_inc: number;
  term: 36 | 60;
  emp_length: number;
  home_ownership: "RENT" | "OWN" | "MORTGAGE";
  fico_range_low: number;
  fico_range_high: number;
  dti: number;
  application_type: "Individual" | "Joint";
  inq_last_6mths: number;
  delinq_2yrs: number;
  acc_now_delinq: number;
  collections_12_mths_ex_med: number;
  chargeoff_within_12_mths: number;
  pub_rec: number;
  earliest_cr_line: string;
};

const defaults: FormValues = {
  loan_amnt: 15000,
  purpose: "debt_consolidation",
  annual_inc: 75000,
  term: 36,
  emp_length: 5,
  home_ownership: "MORTGAGE",
  fico_range_low: 700,
  fico_range_high: 720,
  dti: 18,
  application_type: "Individual",
  inq_last_6mths: 1,
  delinq_2yrs: 0,
  acc_now_delinq: 0,
  collections_12_mths_ex_med: 0,
  chargeoff_within_12_mths: 0,
  pub_rec: 0,
  earliest_cr_line: "2015-06",
};

const labelCls = "block text-xs font-semibold text-slate-700 mb-1.5";
const inputCls = "w-full h-11 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 shadow-sm focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 outline-none transition";

const numberFields: (keyof FormValues)[] = [
  "loan_amnt", "annual_inc", "emp_length", "fico_range_low", "fico_range_high", "dti",
  "inq_last_6mths", "delinq_2yrs", "acc_now_delinq", "collections_12_mths_ex_med",
  "chargeoff_within_12_mths", "pub_rec",
];

export function LoanForm({ loading, onSubmit }: { loading: boolean; onSubmit: (v: LoanInput) => void }) {
  const { register, handleSubmit, control } = useForm<FormValues>({ defaultValues: defaults });

  const submit = (raw: FormValues) => {
    const v = { ...raw } as Record<string, unknown>;
    for (const k of numberFields) v[k] = Number((raw as Record<string, unknown>)[k]) || 0;
    v.term = Number(raw.term) === 60 ? 60 : 36;
    onSubmit(v as unknown as LoanInput);
  };

  return (
    <form onSubmit={handleSubmit(submit)} className="rounded-2xl border border-slate-200 bg-white ll-shadow-soft p-6 sm:p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Loan Application</h2>
          <p className="text-sm text-slate-500 mt-1">17 fields · all data stays in your browser</p>
        </div>
        <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-blue-50 border border-blue-200 px-2.5 py-1 text-[11px] font-semibold text-blue-700">
          <span className="h-1.5 w-1.5 rounded-full bg-blue-600" /> Secure
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div>
          <label className={labelCls}>Loan Amount ($)</label>
          <input type="number" step="100" {...register("loan_amnt")} className={inputCls} />
        </div>
        <div>
          <label className={labelCls}>Loan Purpose</label>
          <select {...register("purpose")} className={inputCls}>
            {PURPOSES.map((p) => <option key={p} value={p}>{purposeLabel(p)}</option>)}
          </select>
        </div>

        <div>
          <label className={labelCls}>Annual Income ($)</label>
          <input type="number" step="500" {...register("annual_inc")} className={inputCls} />
        </div>
        <div>
          <label className={labelCls}>Loan Term</label>
          <Controller
            control={control}
            name="term"
            render={({ field }) => (
              <div className="grid grid-cols-2 rounded-lg border border-slate-200 bg-slate-50 p-1 h-11">
                {[36, 60].map((t) => (
                  <button
                    type="button"
                    key={t}
                    onClick={() => field.onChange(t as 36 | 60)}
                    className={`rounded-md text-sm font-semibold transition ${field.value === t ? "bg-white text-blue-700 shadow-sm" : "text-slate-500"}`}
                  >
                    {t} months
                  </button>
                ))}
              </div>
            )}
          />
        </div>

        <div>
          <label className={labelCls}>
            Employment Length:{" "}
            <Controller control={control} name="emp_length" render={({ field }) => (
              <span className="text-blue-700 font-bold">{field.value}{field.value === 10 ? "+" : ""} yrs</span>
            )} />
          </label>
          <Controller
            control={control}
            name="emp_length"
            render={({ field }) => (
              <input
                type="range" min={0} max={10} step={1}
                value={field.value}
                onChange={(e) => field.onChange(Number(e.target.value))}
                className="w-full h-11 accent-blue-600"
              />
            )}
          />
        </div>
        <div>
          <label className={labelCls}>Home Ownership</label>
          <Controller
            control={control}
            name="home_ownership"
            render={({ field }) => (
              <div className="grid grid-cols-3 rounded-lg border border-slate-200 bg-slate-50 p-1 h-11">
                {(["RENT", "OWN", "MORTGAGE"] as const).map((opt) => (
                  <button type="button" key={opt} onClick={() => field.onChange(opt)}
                    className={`rounded-md text-xs font-semibold transition ${field.value === opt ? "bg-white text-blue-700 shadow-sm" : "text-slate-500"}`}>
                    {opt}
                  </button>
                ))}
              </div>
            )}
          />
        </div>

        <div>
          <label className={labelCls}>FICO Score Low</label>
          <input type="number" {...register("fico_range_low")} className={inputCls} />
        </div>
        <div>
          <label className={labelCls}>FICO Score High</label>
          <input type="number" {...register("fico_range_high")} className={inputCls} />
        </div>

        <div>
          <label className={labelCls}>Debt-to-Income (%)</label>
          <input type="number" step="0.1" {...register("dti")} className={inputCls} />
        </div>
        <div>
          <label className={labelCls}>Application Type</label>
          <Controller
            control={control}
            name="application_type"
            render={({ field }) => (
              <div className="grid grid-cols-2 rounded-lg border border-slate-200 bg-slate-50 p-1 h-11">
                {(["Individual", "Joint"] as const).map((opt) => (
                  <button type="button" key={opt} onClick={() => field.onChange(opt)}
                    className={`rounded-md text-sm font-semibold transition ${field.value === opt ? "bg-white text-blue-700 shadow-sm" : "text-slate-500"}`}>
                    {opt}
                  </button>
                ))}
              </div>
            )}
          />
        </div>

        <div>
          <label className={labelCls}>Credit Inquiries (last 6mo)</label>
          <input type="number" {...register("inq_last_6mths")} className={inputCls} />
        </div>
        <div>
          <label className={labelCls}>Late Payments (last 2yrs)</label>
          <input type="number" {...register("delinq_2yrs")} className={inputCls} />
        </div>

        <div>
          <label className={labelCls}>Accounts Delinquent</label>
          <input type="number" {...register("acc_now_delinq")} className={inputCls} />
        </div>
        <div>
          <label className={labelCls}>Collections (12mo)</label>
          <input type="number" {...register("collections_12_mths_ex_med")} className={inputCls} />
        </div>

        <div>
          <label className={labelCls}>Charge-offs (12mo)</label>
          <input type="number" {...register("chargeoff_within_12_mths")} className={inputCls} />
        </div>
        <div>
          <label className={labelCls}>Public Records</label>
          <input type="number" {...register("pub_rec")} className={inputCls} />
        </div>

        <div className="md:col-span-2">
          <label className={labelCls}>Earliest Credit Line</label>
          <input type="month" {...register("earliest_cr_line")} className={inputCls + " md:max-w-xs"} />
        </div>
      </div>

      <motion.button
        whileHover={!loading ? { scale: 1.01 } : undefined}
        whileTap={!loading ? { scale: 0.99 } : undefined}
        type="submit"
        disabled={loading}
        className="mt-7 group relative w-full h-12 rounded-xl ll-gradient-cta text-white font-semibold text-base shadow-lg shadow-blue-600/30 hover:shadow-blue-600/50 transition-all overflow-hidden disabled:opacity-90"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
        <span className="relative inline-flex items-center justify-center gap-2">
          {loading ? (<><Loader2 className="h-5 w-5 animate-spin" /> Running ML models...</>) : (<>Analyze My Loan <ArrowRight className="h-5 w-5" /></>)}
        </span>
      </motion.button>
    </form>
  );
}
