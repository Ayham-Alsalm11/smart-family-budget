/** The chat page (docs/ARCHITECTURE.md §8, §10): history, sending, the waiting indicator, replies and alerts. */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { saveToken } from "../auth/session";
import { apiError, deferred, json, sent, stubFetch } from "../test/http";
import ChatPage from "./ChatPage";

const REPLY = {
  reply: "تم تسجيل **5,000** على بنزين ومواصلات.",
  steps: [{ step: 1, tool: "add_expense", arguments: { amount: 5000 }, status: "ok", latency_ms: 12 }],
  alerts: [
    { category: "بنزين ومواصلات", status: "warning", spent: 62000, limit: 75000, percent_used: 82.7 },
    { category: "ترفيه", status: "over", spent: 30000, limit: 25000, percent_used: 120 },
  ],
  message_id: 2,
};

function openChat(history: unknown[] = []) {
  saveToken("t");
  const answer = deferred();
  const fake = stubFetch((url) => (url.includes("/messages") ? json(200, { messages: history }) : answer.promise));
  render(
    <MemoryRouter>
      <ChatPage />
    </MemoryRouter>,
  );
  return { fake, answer };
}

describe("chat", () => {
  it("shows the earlier messages", async () => {
    openChat([
      { id: 1, role: "user", content: "كم صرفنا؟", tool_summary: null, created_at: "2026-11-17T10:00:00" },
      { id: 2, role: "assistant", content: "صرفتوا **8,000**.", tool_summary: "get_total → ok", created_at: "2026-11-17T10:00:05" },
    ]);
    expect(await screen.findByText("كم صرفنا؟")).toBeInTheDocument();
    expect(screen.getByText("8,000").tagName).toBe("STRONG");
  });

  it("offers three examples when there is no conversation, and one fills the box", async () => {
    openChat();
    const example = await screen.findByRole("button", { name: "كم صرفنا هالشهر؟" });
    expect(screen.getAllByRole("button", { name: /دفعت 5000 بنزين|كم صرفنا هالشهر؟|كيف بوفّر هالشهر؟/ })).toHaveLength(3);
    await userEvent.click(example);
    expect(screen.getByLabelText("رسالتك")).toHaveValue("كم صرفنا هالشهر؟");
    expect(screen.getByText("16/1000")).toBeInTheDocument();
  });

  it("waits with an indicator, then shows the reply with its alerts and steps", async () => {
    const { fake, answer } = openChat();
    await screen.findByRole("button", { name: "كم صرفنا هالشهر؟" });
    await userEvent.type(screen.getByLabelText("رسالتك"), "دفعت 5000 بنزين{Enter}");

    expect(screen.getByText("ميزان يكتب…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "إرسال" })).toBeDisabled();
    expect(screen.getByText("دفعت 5000 بنزين")).toBeInTheDocument();
    expect(sent(fake, 1).body).toEqual({ message: "دفعت 5000 بنزين" });

    answer.release(json(200, REPLY));
    expect(await screen.findByText("5,000")).toBeInTheDocument();
    expect(screen.queryByText("ميزان يكتب…")).not.toBeInTheDocument();
    const bars = screen.getAllByText(/ميزانية/).filter((element) => element.dataset.status);
    expect(bars.map((bar) => bar.dataset.status)).toEqual(["warning", "over"]);
    expect(bars[0]).toHaveTextContent("اقتربتم من حد ميزانية بنزين ومواصلات: 62,000 من 75,000 (83%)");
    expect(bars[1]).toHaveTextContent("تجاوزتم ميزانية ترفيه");
    expect(screen.getByRole("button", { name: /خطوات الوكيل \(1\)/ })).toBeInTheDocument();
  });

  it("Shift+Enter starts a new line instead of sending", async () => {
    const { fake } = openChat();
    await screen.findByRole("button", { name: "كم صرفنا هالشهر؟" });
    await userEvent.type(screen.getByLabelText("رسالتك"), "سطر{Shift>}{Enter}{/Shift}تاني");
    expect(screen.getByLabelText("رسالتك")).toHaveValue("سطر\nتاني");
    expect(fake).toHaveBeenCalledTimes(1);
  });

  it("shows the agent's refusal message when it is stopped", async () => {
    const { answer } = openChat();
    await screen.findByRole("button", { name: "كم صرفنا هالشهر؟" });
    await userEvent.type(screen.getByLabelText("رسالتك"), "مرحبا{Enter}");
    answer.release(apiError(503, "not_available", "المحادثة متوقفة"));
    expect(await screen.findByRole("alert")).toHaveTextContent("المساعد متوقف حالياً.");
    await waitFor(() => expect(screen.getByLabelText("رسالتك")).toBeEnabled());
  });
});
