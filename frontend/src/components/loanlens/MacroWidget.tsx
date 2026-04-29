import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, TrendingUp, TrendingDown, Minus, Info } from "lucide-react";
import { getMacro, type MacroResponse, type Indicator } from "@/lib/api";

const climateStyles = {
  green: { bar: "from-emerald-500 to-emerald-600", chip: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  amber: { bar: "from-amber-500 to-orange-500",   chip: "bg-amber-50 text-amber-800 border-amber-200" },
  red:   { bar: "from-red-500 to-rose-600",       chip: "bg-red-50 text-red-700 border-red-200" },
};

const toneStyles: Record<Indicator["tone"], string> = {
  good: "bg-emerald-50 border-emerald-200 text-emerald-900",
  warn: "bg-amber-50 border-amber-200 text-amber-900",
  bad:  "bg-red-50 border-red-200 text-red-900",
};

function TrendIcon({ trend }: { trend: Indicator["trend"] }) {
  const cls = "h-4 w-4";
  if (trend === "up") return <TrendingUp className={cls} />;
  if (trend === "down") return <TrendingDown className={cls} />;
  return <Minus className={cls} />;
}

export function MacroWidget({ title }: { title?: string }) {
  const [data, setData] = useState<MacroResponse | null>(null);
  const [hidden, setHidden] = useState(false);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    let alive = true;
    getMacro()
      .then((d) => alive && setData(d))
      .catch(() => alive && setHidden(true));
    return () => { alive = false; };
  }, []);

  if (hidden) return null;
  if (!data) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white ll-shadow-soft p-6 animate-pulse h-40" />
    );
  }

  const c = climateStyles[data.climate.color];
  const tiles = Object.values(data.indicators);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white ll-shadow-soft overflow-hidden">
      {title && (
        <div className="px-6 pt-5 pb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">{title}</div>
      )}
      <div className={`bg-gradient-to-r ${c.bar} text-white px-6 py-4 flex items-center justify-between`}>
        <div className="flex items-center gap-3">
          <div className="text-2xl">{data.climate.emoji}</div>
          <div>
            <div className="text-sm font-bold leading-tight">{data.climate.label}</div>
            <div className="text-xs text-white/85 leading-tight mt-0.5">{data.climate.message}</div>
          </div>
        </div>
        <button
          onClick={() => setOpen((v) => !v)}
          className="rounded-md bg-white/15 hover:bg-white/25 transition px-2 py-1"
          aria-label="Toggle indicators"
        >
          <ChevronDown className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} />
        </button>
      </div>

      <div className="px-6 py-4">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 mb-2">
          <Info className="h-3.5 w-3.5 text-blue-600" />
          How this affects your application
        </div>
        <ul className="space-y-1.5 text-sm text-slate-600">
          {data.impact_on_loan.map((line) => (
            <li key={line} className="flex gap-2">
              <span className="text-blue-600 mt-0.5">•</span><span>{line}</span>
            </li>
          ))}
        </ul>
      </div>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden border-t border-slate-100"
          >
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 p-4">
              {tiles.map((t, i) => (
                <motion.div
                  key={t.label}
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.04 }}
                  className={`rounded-xl border p-3 ${toneStyles[t.tone]}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="text-[11px] font-semibold uppercase tracking-wide opacity-70">{t.label}</div>
                    <TrendIcon trend={t.trend} />
                  </div>
                  <div className="mt-2 text-2xl font-bold tabular-nums">{t.value}</div>
                  <div className="text-[11px] mt-1 opacity-75">{t.note}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
