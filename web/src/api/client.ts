/**
 * The only file that talks to the backend (docs/ARCHITECTURE.md §8, §10, decision A): one function per path.
 *
 * Every request carries the token. The API's one error shape, {"error": {"code", "message", "fields"}}, becomes an
 * ApiError whose `message` is Arabic text for the user. A 401 on a request that carried a token means the session is
 * over: the token is cleared and the app goes back to the login page.
 */
import { clearToken, getToken } from "../auth/session";

export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export type Role = "manager" | "member";
export type Currency = "SYP_OLD" | "SYP_NEW" | "USD";

export interface Member {
  id: number;
  name: string;
  username: string;
  role: Role;
  is_active: boolean;
  created_at: string;
}

export interface Family {
  id: number;
  name: string;
  monthly_income: number;
  base_currency: Currency;
  usd_rate: number | null;
}

export interface LoginReply {
  access_token: string;
  token_type: string;
  member: Member;
}

export interface Profile {
  member: Member;
  family: Family;
}

/** One tool call of a turn, as `/chat` returns it in `steps`. */
export interface Step {
  step: number;
  tool: string;
  arguments: unknown;
  status: string;
  latency_ms: number;
}

/** A budget alert after an expense: `warning` from 80 % of the limit, `over` from 100 %. */
export interface BudgetAlert {
  category: string;
  status: "warning" | "over";
  spent: number;
  limit: number;
  percent_used: number | null;
}

/** An edit or deletion waiting for the member's confirmation (§6.5). */
export interface PendingAction {
  confirmation_id: number;
  action: "edit_expense" | "delete_expense";
  expense: {
    id: number;
    amount: number;
    currency: Currency;
    category: string;
    expense_date: string;
    description: string;
  };
  changes: Record<string, unknown> | null;
}

export interface ChatReply {
  reply: string;
  steps: Step[];
  alerts: BudgetAlert[];
  message_id: number;
  pending_action?: PendingAction;
}

export interface StoredMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  tool_summary: string | null;
  created_at: string;
}

export interface ActionResult {
  id: number;
  status: "confirmed" | "cancelled";
  action: PendingAction["action"];
}

export interface RegisterFamilyInput {
  family_name: string;
  monthly_income: number;
  base_currency: Currency;
  name: string;
  username: string;
  password: string;
}

export interface JoinInput {
  code: string;
  name: string;
  username: string;
  password: string;
}

export const MESSAGES = {
  network: "تعذّر الاتصال بالخادم.",
  sessionOver: "انتهت جلستك، سجّل الدخول من جديد.",
  badLogin: "اسم المستخدم أو كلمة المرور غير صحيحة.",
  forbidden: "هذه المعلومة متاحة لمسؤول العائلة فقط.",
  locked: "تم قفل الدخول مؤقتاً.",
  unavailable: "المساعد متوقف حالياً.",
  unexpected: "حدث خطأ غير متوقع، حاول مرة أخرى.",
};

/** Arabic names of the request fields, for a 422 whose server message is not already Arabic. */
const FIELD_NAMES: Record<string, string> = {
  family_name: "اسم العائلة",
  monthly_income: "الدخل الشهري",
  base_currency: "عملة الأساس",
  name: "الاسم الظاهر",
  username: "اسم المستخدم",
  password: "كلمة المرور",
  code: "كود الدعوة",
  message: "الرسالة",
};

const ARABIC = /[؀-ۿ]/;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly fields: string[];
  /** The server's own text, kept apart from the user's message: the confirmation card reads it (409). */
  readonly serverMessage: string;

  constructor(status: number, code: string, message: string, fields: string[] = [], serverMessage = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.fields = fields;
    this.serverMessage = serverMessage;
  }
}

/**
 * The text the user sees for an error. `withToken` separates a session that ended (401 on a normal request) from a
 * failed login (401 on the login page, which sends no token).
 */
export function userMessage(status: number, serverMessage: string, fields: string[], withToken: boolean): string {
  const arabic = ARABIC.test(serverMessage) ? serverMessage : "";
  switch (status) {
    case 0:
      return MESSAGES.network;
    case 401:
      return withToken ? MESSAGES.sessionOver : arabic || MESSAGES.badLogin;
    case 403:
      return MESSAGES.forbidden;
    case 409:
      return serverMessage || MESSAGES.unexpected;
    case 422: {
      if (arabic) return arabic;
      const names = fields.map((field) => FIELD_NAMES[field] ?? field);
      return names.length ? `تحقّق من هذه الحقول: ${names.join("، ")}.` : "بعض الحقول ناقصة أو غير صحيحة.";
    }
    case 429:
      return arabic ? `${MESSAGES.locked} ${arabic}` : MESSAGES.locked;
    case 503:
      return MESSAGES.unavailable;
    default:
      return arabic || MESSAGES.unexpected;
  }
}

let onUnauthorized: () => void = () => {};

/** The app sets this once, to go back to the login page when a session ends. */
export function setUnauthorizedHandler(handler: () => void): void {
  onUnauthorized = handler;
}

interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
}

async function request<T>(path: string, { method = "GET", body }: RequestOptions = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers.Authorization = `Bearer ${token}`;

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "network", MESSAGES.network);
  }

  const data = await response.json().catch(() => null);
  if (response.ok) return data as T;

  const error = (data && typeof data === "object" && "error" in data ? data.error : {}) as {
    code?: string;
    message?: string;
    fields?: string[];
  };
  const serverMessage = error.message ?? "";
  const fields = error.fields ?? [];
  if (response.status === 401 && token) {
    clearToken();
    onUnauthorized();
  }
  throw new ApiError(response.status, error.code ?? "error", userMessage(response.status, serverMessage, fields, Boolean(token)),
    fields, serverMessage);
}

export function registerFamily(input: RegisterFamilyInput): Promise<LoginReply> {
  return request("/auth/register-family", { method: "POST", body: input });
}

export function joinFamily(input: JoinInput): Promise<LoginReply> {
  return request("/auth/join", { method: "POST", body: input });
}

export function login(username: string, password: string): Promise<LoginReply> {
  return request("/auth/login", { method: "POST", body: { username, password } });
}

export function getMe(): Promise<Profile> {
  return request("/me");
}

export function sendMessage(message: string): Promise<ChatReply> {
  return request("/chat", { method: "POST", body: { message } });
}

export async function getMessages(limit = 50): Promise<StoredMessage[]> {
  const data = await request<{ messages: StoredMessage[] }>(`/messages?limit=${limit}`);
  return data.messages;
}

export function confirmAction(id: number): Promise<ActionResult> {
  return request(`/actions/${id}/confirm`, { method: "POST" });
}

export function cancelAction(id: number): Promise<ActionResult> {
  return request(`/actions/${id}/cancel`, { method: "POST" });
}
