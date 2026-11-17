/** A fake fetch for the tests (docs/ARCHITECTURE.md §10): no backend and no Groq, ever. */
import { vi } from "vitest";

export function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

export function apiError(status: number, code: string, message: string, fields?: string[]): Response {
  return json(status, { error: { code, message, ...(fields ? { fields } : {}) } });
}

type Handler = (url: string, init: RequestInit) => Response | Promise<Response>;

/** Replaces fetch; each call goes to `handler` with the URL and options, and the mock records the calls. */
export function stubFetch(handler: Handler) {
  const fake = vi.fn((input: RequestInfo | URL, init: RequestInit = {}) => Promise.resolve(handler(String(input), init)));
  vi.stubGlobal("fetch", fake);
  return fake;
}

/** What the n-th fetch call sent: the URL, the options and the JSON body. */
export function sent(fake: ReturnType<typeof stubFetch>, index: number) {
  const [url, init = {}] = fake.mock.calls[index] ?? [];
  return { url: String(url), init: init as RequestInit, body: init?.body ? JSON.parse(String(init.body)) : undefined };
}

/** A response the test releases when it wants, to see the page while it waits. */
export function deferred() {
  let release!: (response: Response) => void;
  const promise = new Promise<Response>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}
