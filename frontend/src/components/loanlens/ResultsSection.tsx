import { forwardRef } from "react";
import { motion } from "framer-motion";
import { RotateCcw } from "lucide-react";
import type { LoanInput, PredictResponse } from "@/lib/api";
import { ScoreGauge } from "./ScoreGauge";
import { ShapChart } from "./ShapChart";
import { MacroWidget } from "./MacroWidget";

export const ResultsSection = forwardRef<HTMLDivElement, {
  result: PredictResponse;
  input: LoanInput;
  onReset: () => void;
}>(({ result, input, onReset }, ref) => {
  return (
    <section ref={ref} className="scroll-mt-24">
      <motion.div
        initial="hidden" animate="visible"
        variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.12 } } }}
      >
        <motion.div
          variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}
          className="text-center mb-8"
        >
          <div className="text-xs font-semibold uppercase tracking-wider text-blue-600">Your decision</div>
          <h2 className="mt-2 text-3xl sm:text-4xl font-bold text-slate-900">Results</h2>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}>
            <ScoreGauge result={result} input={input} />
          </motion.div>
          <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}>
            <ShapChart shap={result.shap_values} />
          </motion.div>
        </div>

        <motion.div
          variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}
          className="mt-6"
        >
          <MacroWidget title="Economic Context at Time of Application" />
        </motion.div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={onReset}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 transition"
          >
            <RotateCcw className="h-4 w-4" />
            Start New Application
          </button>
        </div>
      </motion.div>
    </section>
  );
});
ResultsSection.displayName = "ResultsSection";
