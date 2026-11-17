/**
 * The confirmation card of an edit or a deletion (docs/ARCHITECTURE.md §6.5, §10). The buttons call
 * POST /actions/{id}/confirm or /cancel; afterwards the card is disabled and shows the outcome. A 409 means the action
 * is no longer pending: it expired after 10 minutes, or the member already answered «اي» or «لا» in the chat, which
 * stays possible.
 */
import { useState } from "react";

import { ApiError, cancelAction, confirmAction, MESSAGES, type PendingAction } from "../api/client";
import { CURRENCY_NAMES, formatAmount } from "./format";

const FIELD_NAMES: Record<string, string> = {
  amount: "المبلغ",
  category: "الفئة",
  currency: "العملة",
  date: "التاريخ",
  description: "الوصف",
};

/** The server's 409 text names the action's state ("action #5 is already expired"). */
function conflictMessage(serverMessage: string): string {
  if (serverMessage.includes("confirmed")) return "هذا الطلب نُفّذ من قبل.";
  if (serverMessage.includes("cancelled")) return "هذا الطلب أُلغي من قبل.";
  return "انتهت مهلة التأكيد.";
}

function changeValue(field: string, value: unknown): string {
  if (field === "amount" && typeof value === "number") return formatAmount(value);
  if (field === "currency" && typeof value === "string") return CURRENCY_NAMES[value as keyof typeof CURRENCY_NAMES] ?? value;
  return String(value);
}

type State = { phase: "open" | "busy" } | { phase: "done"; text: string; ok: boolean };

export default function ConfirmationCard({ pending }: { pending: PendingAction }) {
  const [state, setState] = useState<State>({ phase: "open" });
  const [failure, setFailure] = useState("");
  const deleting = pending.action === "delete_expense";
  const { expense } = pending;

  async function answer(confirm: boolean) {
    setState({ phase: "busy" });
    setFailure("");
    try {
      if (confirm) {
        await confirmAction(pending.confirmation_id);
        setState({ phase: "done", ok: true, text: deleting ? "تم الحذف ✅" : "تم التعديل ✅" });
      } else {
        await cancelAction(pending.confirmation_id);
        setState({ phase: "done", ok: false, text: "أُلغي الطلب، وما تغيّر شي." });
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setState({ phase: "done", ok: false, text: conflictMessage(error.serverMessage) });
      } else {
        // A network error or similar: the action may still be pending, so the buttons come back.
        setFailure(error instanceof ApiError ? error.message : MESSAGES.unexpected);
        setState({ phase: "open" });
      }
    }
  }

  const disabled = state.phase !== "open";
  return (
    <section
      aria-label={deleting ? "تأكيد الحذف" : "تأكيد التعديل"}
      className={`mt-3 rounded-xl p-3 text-sm ring-1 ${disabled ? "bg-slate-50 ring-slate-200" : "bg-white ring-teal-300"}`}
    >
      <p className="font-semibold">{deleting ? "تأكيد حذف المصروف" : "تأكيد تعديل المصروف"}</p>
      <p className="mt-1 text-slate-700">
        {formatAmount(expense.amount)} {CURRENCY_NAMES[expense.currency] ?? expense.currency} · {expense.category} ·{" "}
        <span dir="ltr">{expense.expense_date}</span>
        {expense.description && ` · ${expense.description}`}
      </p>
      {!deleting && pending.changes && (
        <ul className="mt-1 list-disc ps-5 text-slate-700">
          {Object.entries(pending.changes).map(([field, value]) => (
            <li key={field}>
              {FIELD_NAMES[field] ?? field} ← {changeValue(field, value)}
            </li>
          ))}
        </ul>
      )}
      {state.phase === "done" ? (
        <p role="status" className={`mt-2 font-semibold ${state.ok ? "text-green-700" : "text-slate-600"}`}>
          {state.text}
        </p>
      ) : (
        <>
          <p className="mt-2 text-xs text-slate-500">ينتهي هذا الطلب خلال 10 دقائق. يمكنك أيضاً الرد بـ «اي» أو «لا».</p>
          {failure && <p role="alert" className="mt-2 text-red-700">{failure}</p>}
          <div className="mt-2 flex gap-2">
            <button type="button" disabled={disabled} onClick={() => answer(true)}
              className="rounded-lg bg-teal-700 px-4 py-1.5 font-semibold text-white hover:bg-teal-800 disabled:opacity-60">
              تأكيد
            </button>
            <button type="button" disabled={disabled} onClick={() => answer(false)}
              className="rounded-lg px-4 py-1.5 font-semibold text-slate-700 ring-1 ring-slate-300 hover:bg-slate-100 disabled:opacity-60">
              إلغاء
            </button>
          </div>
        </>
      )}
    </section>
  );
}
