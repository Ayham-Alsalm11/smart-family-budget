/**
 * Budget alerts returned with a reply, as coloured bars above it (docs/ARCHITECTURE.md §7.1, §10): amber from 80 % of
 * a category's limit, red from 100 %.
 */
import type { BudgetAlert } from "../api/client";
import { formatAmount } from "./format";

export default function AlertBar({ alert }: { alert: BudgetAlert }) {
  const over = alert.status === "over";
  const used = alert.percent_used === null ? "" : ` (${Math.round(alert.percent_used)}%)`;
  return (
    <div
      role="status"
      data-status={alert.status}
      className={`mb-2 rounded-xl px-3 py-2 text-sm font-semibold ${
        over ? "bg-red-50 text-red-800 ring-1 ring-red-200" : "bg-amber-50 text-amber-900 ring-1 ring-amber-200"
      }`}
    >
      {over ? "تجاوزتم ميزانية" : "اقتربتم من حد ميزانية"} {alert.category}: {formatAmount(alert.spent)} من{" "}
      {formatAmount(alert.limit)}
      {used}
    </div>
  );
}
