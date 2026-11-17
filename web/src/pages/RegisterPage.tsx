/**
 * Create a family and its first account, a manager (docs/ARCHITECTURE.md §5 flow 1, §10). The rules of §5 are checked
 * here first, and the server's own answer is shown if it refuses.
 */
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router";

import { ApiError, MESSAGES, registerFamily, type Currency } from "../api/client";
import { checkIncome, checkName, checkNewPassword, checkUsername, collectErrors } from "../auth/rules";
import { saveToken } from "../auth/session";
import AuthCard from "../components/AuthCard";
import Field from "../components/Field";
import FormError from "../components/FormError";
import { CURRENCY_NAMES } from "../components/format";
import SubmitButton from "../components/SubmitButton";

export default function RegisterPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    family_name: "", monthly_income: "", base_currency: "SYP_NEW" as Currency, name: "", username: "", password: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [failure, setFailure] = useState("");
  const [busy, setBusy] = useState(false);

  function set(field: keyof typeof form) {
    return (value: string) => setForm((current) => ({ ...current, [field]: value }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const found = collectErrors({
      family_name: checkName(form.family_name, "اسم العائلة", 100),
      monthly_income: checkIncome(form.monthly_income),
      name: checkName(form.name, "الاسم الظاهر", 50),
      username: checkUsername(form.username),
      password: checkNewPassword(form.password),
    });
    setErrors(found);
    setFailure("");
    if (Object.keys(found).length) return;
    setBusy(true);
    try {
      const reply = await registerFamily({
        ...form, family_name: form.family_name.trim(), name: form.name.trim(), username: form.username.trim(),
        monthly_income: Number(form.monthly_income),
      });
      saveToken(reply.access_token);
      navigate("/chat", { replace: true });
    } catch (error) {
      setFailure(error instanceof ApiError ? error.message : MESSAGES.unexpected);
      setBusy(false);
    }
  }

  return (
    <AuthCard
      title="إنشاء عائلة"
      footer={<p>عندك حساب؟ <Link className="font-semibold text-teal-700" to="/login">سجّل الدخول</Link></p>}
    >
      <form onSubmit={submit} noValidate>
        <FormError message={failure} />
        <Field label="اسم العائلة" value={form.family_name} error={errors.family_name}
          onChange={(e) => set("family_name")(e.target.value)} />
        <Field label="الدخل الشهري" type="number" inputMode="decimal" min={0} dir="ltr" value={form.monthly_income}
          error={errors.monthly_income} onChange={(e) => set("monthly_income")(e.target.value)} />
        <Field label="عملة الأساس">
          {(id) => (
            <select id={id} value={form.base_currency} onChange={(e) => set("base_currency")(e.target.value)}
              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 focus:border-teal-600">
              {Object.entries(CURRENCY_NAMES).map(([code, name]) => (
                <option key={code} value={code}>{name}</option>
              ))}
            </select>
          )}
        </Field>
        <Field label="الاسم الظاهر" value={form.name} error={errors.name} onChange={(e) => set("name")(e.target.value)} />
        <Field label="اسم المستخدم" dir="ltr" autoComplete="username" value={form.username} error={errors.username}
          onChange={(e) => set("username")(e.target.value)} />
        <Field label="كلمة المرور" type="password" dir="ltr" autoComplete="new-password" value={form.password}
          error={errors.password} onChange={(e) => set("password")(e.target.value)} />
        <SubmitButton busy={busy} label="إنشاء العائلة" />
      </form>
    </AuthCard>
  );
}
