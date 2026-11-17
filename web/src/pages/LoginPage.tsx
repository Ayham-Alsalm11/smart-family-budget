/**
 * Login with username and password (docs/ARCHITECTURE.md §5, §10). The password rules apply when one is chosen, not
 * at login (§5), so only empty fields are refused here.
 */
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router";

import { ApiError, login, MESSAGES } from "../api/client";
import { collectErrors } from "../auth/rules";
import { saveToken } from "../auth/session";
import AuthCard from "../components/AuthCard";
import Field from "../components/Field";
import FormError from "../components/FormError";
import SubmitButton from "../components/SubmitButton";

export default function LoginPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [failure, setFailure] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const found = collectErrors({
      username: username.trim() ? null : "اكتب اسم المستخدم.",
      password: password ? null : "اكتب كلمة المرور.",
    });
    setErrors(found);
    setFailure("");
    if (Object.keys(found).length) return;
    setBusy(true);
    try {
      const reply = await login(username.trim(), password);
      saveToken(reply.access_token);
      navigate("/chat", { replace: true });
    } catch (error) {
      setFailure(error instanceof ApiError ? error.message : MESSAGES.unexpected);
      setBusy(false);
    }
  }

  return (
    <AuthCard
      title="تسجيل الدخول"
      footer={
        <>
          <p>
            عائلة جديدة؟ <Link className="font-semibold text-teal-700" to="/register">أنشئ عائلة</Link>
          </p>
          <p>
            عندك كود دعوة؟ <Link className="font-semibold text-teal-700" to="/join">انضم لعائلتك</Link>
          </p>
        </>
      }
    >
      <form onSubmit={submit} noValidate>
        <FormError message={failure} />
        <Field label="اسم المستخدم" dir="ltr" autoComplete="username" value={username} error={errors.username}
          onChange={(e) => setUsername(e.target.value)} />
        <Field label="كلمة المرور" type="password" dir="ltr" autoComplete="current-password" value={password}
          error={errors.password} onChange={(e) => setPassword(e.target.value)} />
        <SubmitButton busy={busy} label="دخول" />
      </form>
    </AuthCard>
  );
}
