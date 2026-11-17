/**
 * The account rules of §5, checked before sending so that the member sees a clear message at once
 * (docs/ARCHITECTURE.md §5, §10). The server checks them again and stays the authority.
 */
const USERNAME = /^[a-zA-Z0-9_.]{3,30}$/;
const MAX_PASSWORD_BYTES = 72; // bcrypt's limit; an Arabic letter takes 2 bytes
const INVITE_CODE = /^[A-Z0-9]{8}$/;

export function checkUsername(value: string): string | null {
  if (!value.trim()) return "اكتب اسم المستخدم.";
  if (!USERNAME.test(value.trim())) return "اسم المستخدم من 3 إلى 30 حرفاً: أحرف إنكليزية وأرقام و_ و. فقط.";
  return null;
}

export function checkNewPassword(value: string): string | null {
  if (value.length < 8) return "كلمة المرور 8 أحرف على الأقل.";
  if (new TextEncoder().encode(value).length > MAX_PASSWORD_BYTES) return "كلمة المرور أطول من المسموح.";
  return null;
}

export function checkName(value: string, label: string, limit: number): string | null {
  const length = value.trim().length;
  if (length === 0) return `اكتب ${label}.`;
  if (length > limit) return `${label} من 1 إلى ${limit} حرفاً.`;
  return null;
}

export function checkIncome(value: string): string | null {
  if (value.trim() === "") return "اكتب الدخل الشهري.";
  const income = Number(value);
  if (!Number.isFinite(income) || income < 0) return "الدخل الشهري رقم صفر أو أكثر.";
  return null;
}

export function normaliseInviteCode(value: string): string {
  return value.trim().toUpperCase();
}

export function checkInviteCode(value: string): string | null {
  return INVITE_CODE.test(normaliseInviteCode(value)) ? null : "كود الدعوة 8 أحرف وأرقام.";
}

/** The errors that are not null, by field; an empty object means the form can be sent. */
export function collectErrors(checks: Record<string, string | null>): Record<string, string> {
  return Object.fromEntries(Object.entries(checks).filter((entry): entry is [string, string] => entry[1] !== null));
}
