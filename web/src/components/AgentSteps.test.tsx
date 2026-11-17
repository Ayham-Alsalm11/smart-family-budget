/** The agent-steps panel (docs/ARCHITECTURE.md §10): folded at first, then each step with its coloured status. */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { Step } from "../api/client";
import AgentSteps from "./AgentSteps";

const STEPS: Step[] = [
  { step: 1, tool: "list_expenses", arguments: { period: "this_month" }, status: "ok", latency_ms: 8 },
  { step: 2, tool: "delete_expense", arguments: { expense_id: 61 }, status: "confirmation_required", latency_ms: 15 },
  { step: 3, tool: "get_total", arguments: { member: "أم أحمد" }, status: "forbidden", latency_ms: 3 },
];

describe("agent steps", () => {
  it("is folded, then opens to show every step", async () => {
    render(<AgentSteps steps={STEPS} />);
    const toggle = screen.getByRole("button", { name: /خطوات الوكيل \(3\)/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("list_expenses")).not.toBeInTheDocument();

    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("list_expenses")).toBeInTheDocument();
    expect(screen.getByText("ok")).toHaveClass("bg-green-100");
    expect(screen.getByText("confirmation_required")).toHaveClass("bg-yellow-100");
    expect(screen.getByText("forbidden")).toHaveClass("bg-red-100");
    expect(screen.getByText("15 ms")).toBeInTheDocument();
    expect(screen.getByText('{"expense_id":61}')).toBeInTheDocument();

    await userEvent.click(toggle);
    expect(screen.queryByText("list_expenses")).not.toBeInTheDocument();
  });

  it("says when no tool was called", async () => {
    render(<AgentSteps steps={[]} />);
    await userEvent.click(screen.getByRole("button", { name: /خطوات الوكيل \(0\)/ }));
    expect(screen.getByText("لم يستدعِ الوكيل أي أداة.")).toBeInTheDocument();
  });
});
