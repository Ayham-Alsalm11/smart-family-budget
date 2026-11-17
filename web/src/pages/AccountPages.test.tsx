/** Login, registration and joining (docs/ARCHITECTURE.md §5, §10): success, a refusal from the server, and the rules. */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { getToken } from "../auth/session";
import { apiError, json, sent, stubFetch } from "../test/http";
import JoinPage from "./JoinPage";
import LoginPage from "./LoginPage";
import RegisterPage from "./RegisterPage";

const SESSION = { access_token: "new-token", token_type: "bearer", member: { id: 1, name: "أبو أحمد", role: "manager" } };

function open(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/join" element={<JoinPage />} />
        <Route path="/chat" element={<p>صفحة المحادثة</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("login", () => {
  it("saves the token and opens the chat", async () => {
    const fake = stubFetch(() => json(200, SESSION));
    open("/login");
    await userEvent.type(screen.getByLabelText("اسم المستخدم"), "abu.ahmad");
    await userEvent.type(screen.getByLabelText("كلمة المرور"), "choose-a-long-one");
    await userEvent.click(screen.getByRole("button", { name: "دخول" }));
    expect(await screen.findByText("صفحة المحادثة")).toBeInTheDocument();
    expect(getToken()).toBe("new-token");
    expect(sent(fake, 0).body).toEqual({ username: "abu.ahmad", password: "choose-a-long-one" });
  });

  it("shows the server's refusal and stays on the page", async () => {
    stubFetch(() => apiError(401, "unauthorized", "اسم المستخدم أو كلمة المرور غير صحيحة."));
    open("/login");
    await userEvent.type(screen.getByLabelText("اسم المستخدم"), "abu.ahmad");
    await userEvent.type(screen.getByLabelText("كلمة المرور"), "wrong-password");
    await userEvent.click(screen.getByRole("button", { name: "دخول" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("اسم المستخدم أو كلمة المرور غير صحيحة.");
    expect(getToken()).toBeNull();
    expect(screen.getByRole("button", { name: "دخول" })).toBeEnabled();
  });

  it("does not send empty fields", async () => {
    const fake = stubFetch(() => json(200, SESSION));
    open("/login");
    await userEvent.click(screen.getByRole("button", { name: "دخول" }));
    expect(screen.getByText("اكتب اسم المستخدم.")).toBeInTheDocument();
    expect(fake).not.toHaveBeenCalled();
  });
});

describe("create a family", () => {
  it("checks the rules of §5 before sending", async () => {
    const fake = stubFetch(() => json(201, SESSION));
    open("/register");
    await userEvent.type(screen.getByLabelText("اسم المستخدم"), "أبو");
    await userEvent.type(screen.getByLabelText("كلمة المرور"), "short");
    await userEvent.click(screen.getByRole("button", { name: "إنشاء العائلة" }));
    expect(screen.getByText("اكتب اسم العائلة.")).toBeInTheDocument();
    expect(screen.getByText(/اسم المستخدم من 3 إلى 30 حرفاً/)).toBeInTheDocument();
    expect(screen.getByText("كلمة المرور 8 أحرف على الأقل.")).toBeInTheDocument();
    expect(fake).not.toHaveBeenCalled();
  });

  it("sends the family and opens the chat", async () => {
    const fake = stubFetch(() => json(201, SESSION));
    open("/register");
    await userEvent.type(screen.getByLabelText("اسم العائلة"), "عائلة السالم");
    await userEvent.type(screen.getByLabelText("الدخل الشهري"), "500000");
    await userEvent.selectOptions(screen.getByLabelText("عملة الأساس"), "SYP_OLD");
    await userEvent.type(screen.getByLabelText("الاسم الظاهر"), "أبو أحمد");
    await userEvent.type(screen.getByLabelText("اسم المستخدم"), "abu.ahmad");
    await userEvent.type(screen.getByLabelText("كلمة المرور"), "choose-a-long-one");
    await userEvent.click(screen.getByRole("button", { name: "إنشاء العائلة" }));
    expect(await screen.findByText("صفحة المحادثة")).toBeInTheDocument();
    expect(sent(fake, 0).body).toEqual({
      family_name: "عائلة السالم", monthly_income: 500000, base_currency: "SYP_OLD", name: "أبو أحمد",
      username: "abu.ahmad", password: "choose-a-long-one",
    });
  });
});

describe("join with a code", () => {
  it("sends the code in upper case and shows a used code's message", async () => {
    const fake = stubFetch(() => apiError(409, "conflict", "كود الدعوة مستعمل.", ["code"]));
    open("/join");
    await userEvent.type(screen.getByLabelText("كود الدعوة"), "abcd2345");
    await userEvent.type(screen.getByLabelText("الاسم الظاهر"), "أم أحمد");
    await userEvent.type(screen.getByLabelText("اسم المستخدم"), "um.ahmad");
    await userEvent.type(screen.getByLabelText("كلمة المرور"), "another-long-one");
    await userEvent.click(screen.getByRole("button", { name: "انضمام" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("كود الدعوة مستعمل.");
    expect(sent(fake, 0).body.code).toBe("ABCD2345");
  });
});
