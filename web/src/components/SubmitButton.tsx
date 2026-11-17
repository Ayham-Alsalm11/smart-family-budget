/** The main button of a form, disabled while the request runs (docs/ARCHITECTURE.md §10). */
export default function SubmitButton({ busy, label }: { busy: boolean; label: string }) {
  return (
    <button
      type="submit"
      disabled={busy}
      className="w-full rounded-xl bg-teal-700 px-4 py-2.5 font-semibold text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {busy ? "لحظة…" : label}
    </button>
  );
}
