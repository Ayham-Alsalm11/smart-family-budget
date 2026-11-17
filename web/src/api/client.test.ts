/** The API client (docs/ARCHITECTURE.md §8, §10): the token on every request, and one Arabic message per error. */
import { describe, expect, it, vi } from "vitest";

import { apiError, json, sent, stubFetch } from "../test/http";
import { getToken, saveToken } from "../auth/session";
import { API_URL, ApiError, getMe, login, MESSAGES, sendMessage, setUnauthorizedHandler } from "./client";

async function failure(call: () => Promise<unknown>): Promise<ApiError> {
  try {
    await call();
  } catch (error) {
    if (error instanceof ApiError) return error;
    throw error;
  }
  throw new Error("the request did not fail");
}

describe("the token", () => {
  it("is added to every request", async () => {
    saveToken("the-token");
    const fake = stubFetch(() => json(200, { reply: "تمام", steps: [], alerts: [], message_id: 1 }));
    await sendMessage("دفعت 5000 بنزين");
    const { url, init, body } = sent(fake, 0);
    expect(url).toBe(`${API_URL}/chat`);
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer the-token");
    expect(body).toEqual({ message: "دفعت 5000 بنزين" });
  });

  it("is not sent when there is none", async () => {
    const fake = stubFetch(() => json(200, { access_token: "x", token_type: "bearer", member: {} }));
    await login("abu.ahmad", "a-long-password");
    expect((sent(fake, 0).init.headers as Record<string, string>).Authorization).toBeUndefined();
  });
});

describe("error messages", () => {
  it.each([
    [403, "forbidden", "only a manager can create invites", MESSAGES.forbidden],
    [409, "conflict", "اسم المستخدم مأخوذ.", "اسم المستخدم مأخوذ."],
    [429, "locked", "", MESSAGES.locked],
    [503, "not_available", "المحادثة متوقفة", MESSAGES.unavailable],
  ])("%i gives its message", async (status, code, serverMessage, expected) => {
    saveToken("t");
    stubFetch(() => apiError(status, code, serverMessage));
    const error = await failure(getMe);
    expect(error.status).toBe(status);
    expect(error.message).toBe(expected);
  });

  it("422 names the missing fields in Arabic", async () => {
    stubFetch(() => apiError(422, "missing_fields", "username: Field required", ["username", "password"]));
    const error = await failure(() => login("", ""));
    expect(error.message).toBe("تحقّق من هذه الحقول: اسم المستخدم، كلمة المرور.");
    expect(error.fields).toEqual(["username", "password"]);
  });

  it("422 keeps the server's message when it is already Arabic", async () => {
    stubFetch(() => apiError(422, "invalid_args", "كلمة المرور 8 أحرف على الأقل.", ["password"]));
    expect((await failure(() => login("a", "b"))).message).toBe("كلمة المرور 8 أحرف على الأقل.");
  });

  it("429 keeps the server's waiting time", async () => {
    stubFetch(() => apiError(429, "locked", "حاول بعد 5 دقيقة."));
    expect((await failure(() => login("a", "b"))).message).toBe(`${MESSAGES.locked} حاول بعد 5 دقيقة.`);
  });

  it("a network failure says the server cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    const error = await failure(getMe);
    expect(error.status).toBe(0);
    expect(error.message).toBe(MESSAGES.network);
  });
});

describe("401", () => {
  it("clears the token and opens the login page when a session ends", async () => {
    saveToken("expired");
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    stubFetch(() => apiError(401, "unauthorized", "invalid token"));
    const error = await failure(getMe);
    expect(error.message).toBe(MESSAGES.sessionOver);
    expect(getToken()).toBeNull();
    expect(handler).toHaveBeenCalledOnce();
  });

  it("on the login page it is a wrong password, not an ended session", async () => {
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    stubFetch(() => apiError(401, "unauthorized", "اسم المستخدم أو كلمة المرور غير صحيحة."));
    expect((await failure(() => login("abu.ahmad", "wrong-password"))).message).toBe(MESSAGES.badLogin);
    expect(handler).not.toHaveBeenCalled();
  });
});
