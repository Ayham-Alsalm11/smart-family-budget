/** Display helpers shared by the pages (docs/ARCHITECTURE.md §10): currency names in Arabic, and amounts. */
import type { Currency } from "../api/client";

export const CURRENCY_NAMES: Record<Currency, string> = {
  SYP_OLD: "ليرة سورية قديمة",
  SYP_NEW: "ليرة سورية جديدة",
  USD: "دولار أمريكي",
};

/** Western digits with thousands separators, as the agent writes them ("5,000"). */
export function formatAmount(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}
