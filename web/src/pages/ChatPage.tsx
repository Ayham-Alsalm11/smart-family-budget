/**
 * The chat with the agent (docs/ARCHITECTURE.md §6.5, §8, §10), the heart of step 9a.
 *
 * Earlier messages come from GET /messages, as text only: steps, alerts and confirmation cards exist only for the
 * replies of this visit, because the server does not store them with the message. A pending action from an earlier
 * visit can still be answered with «اي» or «لا» in the chat (§6.5).
 */
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { ApiError, getMessages, MESSAGES, sendMessage, type ChatReply } from "../api/client";
import AgentSteps from "../components/AgentSteps";
import AlertBar from "../components/AlertBar";
import ConfirmationCard from "../components/ConfirmationCard";
import MarkdownText from "../components/MarkdownText";

export const MAX_MESSAGE = 1000; // as on the server (§8)
const EXAMPLES = ["دفعت 5000 بنزين", "كم صرفنا هالشهر؟", "كيف بوفّر هالشهر؟"];

type Item =
  | { kind: "user"; key: string; text: string }
  | { kind: "assistant"; key: string; text: string; reply?: ChatReply }
  | { kind: "error"; key: string; text: string };

let nextKey = 0;
const key = () => `local-${nextKey++}`;

export default function ChatPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [draft, setDraft] = useState("");
  const [waiting, setWaiting] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    getMessages()
      .then((messages) => {
        if (!active) return;
        setItems(messages.map((m) => ({ kind: m.role, key: `m-${m.id}`, text: m.content })));
      })
      .catch((error) => active && setLoadError(error instanceof ApiError ? error.message : MESSAGES.unexpected))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    end.current?.scrollIntoView?.({ block: "end" });
  }, [items, waiting]);

  // The box is disabled while a reply is awaited, so it can take the focus back only once it is enabled again.
  useEffect(() => {
    if (!waiting) input.current?.focus();
  }, [waiting]);

  async function send(event?: FormEvent) {
    event?.preventDefault();
    const text = draft.trim();
    if (!text || waiting) return;
    setItems((current) => [...current, { kind: "user", key: key(), text }]);
    setDraft("");
    setWaiting(true);
    try {
      const reply = await sendMessage(text);
      setItems((current) => [...current, { kind: "assistant", key: key(), text: reply.reply, reply }]);
    } catch (error) {
      const message = error instanceof ApiError ? error.message : MESSAGES.unexpected;
      setItems((current) => [...current, { kind: "error", key: key(), text: message }]);
    } finally {
      setWaiting(false);
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends, Shift+Enter starts a new line; Enter that confirms an input-method choice does neither.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void send();
    }
  }

  const empty = !loading && !loadError && items.length === 0;
  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4" aria-live="polite">
        {loading && <p className="text-center text-sm text-slate-500">تحميل المحادثة…</p>}
        {loadError && <p role="alert" className="text-center text-sm text-red-700">{loadError}</p>}
        {empty && (
          <div className="mt-10 text-center">
            <p className="text-lg font-semibold">أهلاً! أنا ميزان، مساعد ميزانية العائلة.</p>
            <p className="mt-1 text-sm text-slate-500">اكتب مصروفاً أو اسأل عن مصاريفكم. مثلاً:</p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {EXAMPLES.map((example) => (
                <button key={example} type="button" onClick={() => { setDraft(example); input.current?.focus(); }}
                  className="rounded-full bg-white px-4 py-1.5 text-sm text-teal-800 ring-1 ring-teal-200 hover:bg-teal-50">
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}
        {items.map((item) => {
          if (item.kind === "user") {
            return (
              <div key={item.key} className="flex justify-start">
                <p data-role="user" className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-ss-sm bg-teal-700 px-4 py-2 text-white">
                  {item.text}
                </p>
              </div>
            );
          }
          if (item.kind === "error") {
            return (
              <p key={item.key} role="alert" className="mx-auto max-w-[85%] rounded-xl bg-red-50 px-4 py-2 text-center text-sm text-red-700">
                {item.text}
              </p>
            );
          }
          return (
            <div key={item.key} className="flex justify-end">
              <div data-role="assistant" className="max-w-[85%] rounded-2xl rounded-se-sm bg-white px-4 py-2 shadow-sm ring-1 ring-slate-200">
                {item.reply?.alerts.map((alert) => <AlertBar key={alert.category} alert={alert} />)}
                <MarkdownText text={item.text} />
                {item.reply?.pending_action && <ConfirmationCard pending={item.reply.pending_action} />}
                {item.reply && <AgentSteps steps={item.reply.steps} />}
              </div>
            </div>
          );
        })}
        {waiting && (
          <div className="flex justify-end">
            <p role="status" className="rounded-2xl bg-white px-4 py-2 text-sm text-slate-500 ring-1 ring-slate-200">
              ميزان يكتب…
            </p>
          </div>
        )}
        <div ref={end} />
      </div>
      <form onSubmit={send} className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="flex items-end gap-2">
          <textarea
            ref={input}
            value={draft}
            maxLength={MAX_MESSAGE}
            rows={2}
            disabled={waiting}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            aria-label="رسالتك"
            placeholder="اكتب رسالتك… (Shift+Enter لسطر جديد)"
            className="min-h-11 flex-1 resize-none rounded-xl border border-slate-300 px-3 py-2 outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100 disabled:bg-slate-50"
          />
          <button type="submit" disabled={waiting || !draft.trim()}
            className="rounded-xl bg-teal-700 px-5 py-2.5 font-semibold text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50">
            إرسال
          </button>
        </div>
        <p className="mt-1 text-xs text-slate-500" dir="ltr">{draft.length}/{MAX_MESSAGE}</p>
      </form>
    </div>
  );
}
