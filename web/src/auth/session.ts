/**
 * The login token (JWT) in localStorage (docs/ARCHITECTURE.md §10, decision C).
 *
 * Acceptable here: React escapes every text it renders and the agent's Markdown is shown without raw HTML, so a script
 * cannot be injected to read the token; and the server checks every permission itself, so the token only says who the
 * member is (§5). Storage can be blocked (private windows), which is why every access is guarded.
 */
const TOKEN_KEY = "mizan.token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function saveToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Without storage the member has to log in again on the next visit; nothing else breaks.
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Nothing was stored.
  }
}
