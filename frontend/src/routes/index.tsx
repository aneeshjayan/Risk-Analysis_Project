import { useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { toast, Toaster } from "sonner";
import { LoanLensHeader } from "@/components/loanlens/Header";
import { Hero } from "@/components/loanlens/Hero";
import { MacroWidget } from "@/components/loanlens/MacroWidget";
import { LoanForm } from "@/components/loanlens/LoanForm";
import { ResultsSection } from "@/components/loanlens/ResultsSection";
import { CopilotFab } from "@/components/loanlens/CopilotFab";
import { predict, type LoanInput, type PredictResponse } from "@/lib/api";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [input, setInput] = useState<LoanInput | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (values: LoanInput) => {
    setLoading(true);
    try {
      const r = await predict(values);
      setResult(r);
      setInput(values);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Prediction failed";
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setInput(null);
    formRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <LoanLensHeader />
      <Hero />

      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 -mt-12 pb-24 space-y-8 relative z-10">
        <MacroWidget />

        <div ref={formRef} className="scroll-mt-24">
          <LoanForm loading={loading} onSubmit={handleSubmit} />
        </div>

        {result && input && (
          <ResultsSection ref={resultsRef} result={result} input={input} onReset={handleReset} />
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div>© {new Date().getFullYear()} LoanLens · AI-powered loan decisions</div>
          <div>Models: XGBoost · LightGBM · Random Forest · SHAP explainability</div>
        </div>
      </footer>

      <CopilotFab
        context={{
          decision:    result?.decision,
          probability: result?.probability,
          risk_tier:   result?.risk_tier,
          best_model:  result?.best_model,
          model_probs: result?.model_probs,
          inputs:      input ?? undefined,
          shap_values: result?.shap_values,
          // raw backend values — passed to GPT so the LLM prompt is accurate
          _raw_default_prob: result?._raw_default_prob,
          _raw_model_probs:  result?._raw_model_probs,
          _raw_shap_dict:    result?._raw_shap_dict,
        }}
      />

      <Toaster position="top-right" richColors closeButton />
    </div>
  );
}
