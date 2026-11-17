/** The confirmation card (docs/ARCHITECTURE.md §6.5, §10): confirm, cancel, and an action that is no longer pending. */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { PendingAction } from "../api/client";
import { saveToken } from "../auth/session";
import { apiError, json, sent, stubFetch } from "../test/http";
import ConfirmationCard from "./ConfirmationCard";

const DELETE: PendingAction = {
  confirmation_id: 7,
  action: "delete_expense",
  expense: { id: 61, amount: 5000, currency: "SYP_OLD", category: "بنزين ومواصلات", expense_date: "2026-11-17", description: "بنزين" },
  changes: null,
};
const EDIT: PendingAction = { ...DELETE, action: "edit_expense", changes: { amount: 6000 } };

describe("confirmation card", () => {
  it("shows the expense and the 10 minutes", () => {
    render(<ConfirmationCard pending={EDIT} />);
    expect(screen.getByText(/5,000 ليرة سورية قديمة · بنزين ومواصلات/)).toBeInTheDocument();
    expect(screen.getByText("المبلغ ← 6,000")).toBeInTheDocument();
    expect(screen.getByText(/ينتهي هذا الطلب خلال 10 دقائق/)).toBeInTheDocument();
  });

  it("confirms, then is disabled with the outcome", async () => {
    saveToken("t");
    const fake = stubFetch(() => json(200, { id: 7, status: "confirmed", action: "delete_expense" }));
    render(<ConfirmationCard pending={DELETE} />);
    await userEvent.click(screen.getByRole("button", { name: "تأكيد" }));
    expect(await screen.findByText("تم الحذف ✅")).toBeInTheDocument();
    expect(sent(fake, 0).url).toMatch(/\/actions\/7\/confirm$/);
    expect(sent(fake, 0).init.method).toBe("POST");
    expect(screen.queryByRole("button", { name: "تأكيد" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "إلغاء" })).not.toBeInTheDocument();
  });

  it("cancels", async () => {
    saveToken("t");
    const fake = stubFetch(() => json(200, { id: 7, status: "cancelled", action: "edit_expense" }));
    render(<ConfirmationCard pending={EDIT} />);
    await userEvent.click(screen.getByRole("button", { name: "إلغاء" }));
    expect(await screen.findByText("أُلغي الطلب، وما تغيّر شي.")).toBeInTheDocument();
    expect(sent(fake, 0).url).toMatch(/\/actions\/7\/cancel$/);
  });

  it("says the time ran out on a 409", async () => {
    saveToken("t");
    stubFetch(() => apiError(409, "conflict", "action #7 is already expired"));
    render(<ConfirmationCard pending={DELETE} />);
    await userEvent.click(screen.getByRole("button", { name: "تأكيد" }));
    expect(await screen.findByText("انتهت مهلة التأكيد.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "تأكيد" })).not.toBeInTheDocument();
  });

  it("says so when «اي» in the chat already ran it", async () => {
    saveToken("t");
    stubFetch(() => apiError(409, "conflict", "action #7 is already confirmed"));
    render(<ConfirmationCard pending={DELETE} />);
    await userEvent.click(screen.getByRole("button", { name: "تأكيد" }));
    expect(await screen.findByText("هذا الطلب نُفّذ من قبل.")).toBeInTheDocument();
  });

  it("keeps the buttons after a network error, because the action may still be pending", async () => {
    saveToken("t");
    stubFetch(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<ConfirmationCard pending={DELETE} />);
    await userEvent.click(screen.getByRole("button", { name: "تأكيد" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("تعذّر الاتصال بالخادم.");
    expect(screen.getByRole("button", { name: "تأكيد" })).toBeEnabled();
  });
});
