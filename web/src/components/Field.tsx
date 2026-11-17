/** A labelled form field with its error under it (docs/ARCHITECTURE.md §10). */
import { useId, type InputHTMLAttributes, type ReactNode } from "react";

interface Props extends Omit<InputHTMLAttributes<HTMLInputElement>, "children"> {
  label: string;
  error?: string;
  /** A replacement for the input, such as a select; it receives the same id. */
  children?: (id: string) => ReactNode;
}

export default function Field({ label, error, children, ...input }: Props) {
  const id = useId();
  const errorId = `${id}-error`;
  return (
    <div className="mb-4">
      <label htmlFor={id} className="mb-1 block text-sm font-semibold text-slate-700">
        {label}
      </label>
      {children ? (
        children(id)
      ) : (
        <input
          id={id}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? errorId : undefined}
          className="w-full rounded-xl border border-slate-300 px-3 py-2 outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
          {...input}
        />
      )}
      {error && (
        <p id={errorId} className="mt-1 text-sm text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}
