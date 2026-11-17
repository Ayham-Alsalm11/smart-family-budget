/** The server's answer when a form is refused, read out by screen readers (docs/ARCHITECTURE.md §10). */
export default function FormError({ message }: { message: string }) {
  if (!message) return null;
  return (
    <p role="alert" className="mb-4 rounded-xl bg-red-50 px-3 py-2 text-sm text-red-700 ring-1 ring-red-200">
      {message}
    </p>
  );
}
