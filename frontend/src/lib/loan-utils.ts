export const PURPOSES = [
  "debt_consolidation",
  "credit_card",
  "home_improvement",
  "major_purchase",
  "medical",
  "small_business",
  "car",
  "vacation",
  "moving",
  "other",
] as const;

export const purposeLabel = (k: string) =>
  k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export const fmtCurrency = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);

export const fmtPct = (n: number, digits = 1) => `${(n * 100).toFixed(digits)}%`;

export const tierColor = (tier: string) => {
  switch (tier) {
    case "Low": return "text-emerald-600 bg-emerald-50 border-emerald-200";
    case "Moderate": return "text-amber-600 bg-amber-50 border-amber-200";
    case "High": return "text-orange-600 bg-orange-50 border-orange-200";
    default: return "text-red-600 bg-red-50 border-red-200";
  }
};
