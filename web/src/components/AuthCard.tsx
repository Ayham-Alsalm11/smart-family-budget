/** The frame of the three account pages (docs/ARCHITECTURE.md §10): the app's name, a title and a centred card. */
import type { ReactNode } from "react";

interface Props {
  title: string;
  children: ReactNode;
  footer: ReactNode;
}

export default function AuthCard({ title, children, footer }: Props) {
  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <p className="text-4xl font-bold text-teal-700">ميزان</p>
          <p className="mt-1 text-sm text-slate-500">مساعد ميزانية العائلة</p>
        </div>
        <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
          <h1 className="mb-5 text-xl font-semibold">{title}</h1>
          {children}
        </section>
        <div className="mt-4 space-y-1 text-center text-sm text-slate-600">{footer}</div>
      </div>
    </main>
  );
}
