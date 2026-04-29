import { motion } from "framer-motion";

export function Hero() {
  return (
    <section className="ll-gradient-header text-white pt-12 pb-20 sm:pt-16 sm:pb-28 relative overflow-hidden">
      <div className="absolute inset-0 opacity-30 pointer-events-none" style={{
        backgroundImage: "radial-gradient(circle at 20% 20%, rgba(59,130,246,0.4) 0, transparent 40%), radial-gradient(circle at 80% 60%, rgba(99,102,241,0.3) 0, transparent 40%)"
      }} />
      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 rounded-full bg-white/10 border border-white/15 px-3 py-1 text-xs font-medium text-blue-100 mb-5"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
          Live ML inference · sub-second decisions
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.05 }}
          className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight"
        >
          <span className="ll-gradient-text">Loan Approval Predictor</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="mt-5 max-w-2xl mx-auto text-base sm:text-lg text-blue-100/80"
        >
          Instant AI-powered decision — XGBoost, LightGBM &amp; Random Forest with full SHAP explanation.
        </motion.p>
      </div>
    </section>
  );
}
