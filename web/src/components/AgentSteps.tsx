/**
 * The agent-steps panel under a reply (docs/ARCHITECTURE.md §8 `steps[]`, §10), folded by default: each tool call with
 * its status in colour, its time and its arguments. It shows the committee what the agent did to answer.
 */
import { useId, useState } from "react";

import type { Step } from "../api/client";

const STATUS_COLOURS: Record<string, string> = {
  ok: "bg-green-100 text-green-800",
  confirmation_required: "bg-yellow-100 text-yellow-800",
  forbidden: "bg-red-100 text-red-800",
  invalid_args: "bg-red-100 text-red-800",
};
const OTHER_STATUS = "bg-slate-100 text-slate-700";

export default function AgentSteps({ steps }: { steps: Step[] }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  return (
    <div className="mt-2 text-sm">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
        className="text-xs font-semibold text-slate-500 hover:text-teal-700"
      >
        {open ? "▾" : "◂"} خطوات الوكيل ({steps.length})
      </button>
      {open && (
        <div id={panelId} className="mt-2 space-y-2">
          {steps.length === 0 && <p className="text-xs text-slate-500">لم يستدعِ الوكيل أي أداة.</p>}
          {steps.map((step, index) => (
            <div key={index} className="rounded-xl bg-slate-50 p-2 ring-1 ring-slate-200">
              <div className="flex flex-wrap items-center gap-2">
                <code dir="ltr" className="font-semibold">{step.tool}</code>
                <span className={`rounded-full px-2 py-0.5 text-xs ${STATUS_COLOURS[step.status] ?? OTHER_STATUS}`}>
                  {step.status}
                </span>
                <span className="text-xs text-slate-500" dir="ltr">{step.latency_ms} ms</span>
              </div>
              <pre dir="ltr" className="mt-1 overflow-x-auto whitespace-pre-wrap break-all text-left text-xs text-slate-600">
                {JSON.stringify(step.arguments)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
