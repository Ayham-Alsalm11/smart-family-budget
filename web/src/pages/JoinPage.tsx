/**
 * Join a family with an invite code from its manager (docs/ARCHITECTURE.md §5 flow 2, §10). The code is accepted in
 * either case, as on the server; the role comes with the code.
 */
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router";

import { ApiError, joinFamily, MESSAGES } from "../api/client";
import { checkInviteCode, checkName, checkNewPassword, checkUsername, collectErrors, normaliseInviteCode } from "../auth/rules";
import { saveToken } from "../auth/session";
import AuthCard from "../components/AuthCard";
import Field from "../components/Field";
import FormError from "../components/FormError";
import SubmitButton from "../components/SubmitButton";

export default function JoinPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ code: "", name: "", username: "", password: "" });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [failure, setFailure] = useState("");
  const [busy, setBusy] = useState(false);

  function set(field: keyof typeof form) {
    return (value: string) => setForm((current) => ({ ...current, [field]: value }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const found = collectErrors({
      code: checkInviteCode(form.code),
      name: checkName(form.name, "الاسم الظاهر", 50),
      username: checkUsername(form.username),
      password: checkNewPassword(form.password),
    });
    setErrors(found);
    setFailure("");
    if (Object.keys(found).length) return;
    setBusy(true);
    try {
      const reply = await joinFamily({
        code: normaliseInviteCode(form.code), name: form.name.trim(), username: form.username.trim(),
        password: form.password,
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
      title="الانضمام بكود دعوة"
      footer={<p>عندك حساب؟ <Link className="font-semibold text-teal-700" to="/login">سجّل الدخول</Link></p>}
    >
      <form onSubmit={submit} noValidate>
        <FormError message={failure} />
        <Field label="كود الدعوة" dir="ltr" autoComplete="off" value={form.code} error={errors.code}
          onChange={(e) => set("code")(e.target.value)} />
        <Field label="الاسم الظاهر" value={form.name} error={errors.name} onChange={(e) => set("name")(e.target.value)} />
        <Field label="اسم المستخدم" dir="ltr" autoComplete="username" value={form.username} error={errors.username}
          onChange={(e) => set("username")(e.target.value)} />
        <Field label="كلمة المرور" type="password" dir="ltr" autoComplete="new-password" value={form.password}
          error={errors.password} onChange={(e) => set("password")(e.target.value)} />
        <SubmitButton busy={busy} label="انضمام" />
      </form>
    </AuthCard>
  );
}
