# Web interface (web)

React interface of the Family Budget Agent ("ميزان"), first version (step 9a). Specification:
[../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) §10.

React + Vite + TypeScript, Tailwind CSS, React Router, react-markdown, and the IBM Plex Sans Arabic font from
`@fontsource` (nothing is loaded from the internet). Arabic, right to left. Every version is pinned in `package.json`,
and `package-lock.json` is committed.

## Setup (Windows, once)

Node.js 26 and npm 11. From `web\`:

```
npm ci
```

The backend address comes from `VITE_API_URL` and defaults to `http://localhost:8000`. To change it, copy
`.env.example` to `.env`. Never put a secret there: every `VITE_` variable ends up in the browser.

## Running with the backend

Two terminals:

1. **API**, from `backend\` (see [../backend/README.md](../backend/README.md); ChromaDB and Ollama are needed only for
   search by meaning):
   ```
   .venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
2. **Web**, from `web\`:
   ```
   npm run dev
   ```
   Open **http://localhost:5173**, not `127.0.0.1:5173`: the API allows only that origin (`CORS_ORIGINS`). If the
   page says "تعذّر الاتصال بالخادم" while the API is running, set `VITE_API_URL=http://127.0.0.1:8000` in `web\.env`
   and start `npm run dev` again.

## Checks

```
npm test
npm run build
```

The tests use a fake `fetch`: no backend and no Groq.

## Manual check

Steps 3–6 send messages to the agent and spend Groq quota.

1. **Create a family:** "أنشئ عائلة" on the login page. After it, the chat opens, with the family, your name and
   "مسؤول" in the top bar.
2. **Invite and join:** in Swagger (http://127.0.0.1:8000/docs), `POST /auth/login` with your account, copy
   `access_token`, press **Authorize** and paste it, then `POST /invites` with `{"role": "member"}` and copy `code`.
   Open a private window at http://localhost:5173/join and join with the code; the top bar says "فرد".
3. **Record an expense** (manager): "دفعت 5000 بنزين". The reply confirms it; open "خطوات الوكيل" and check that
   `add_expense` is green.
4. **Ask** (manager): "كم صرفنا هالشهر؟". The total includes the 5000 (`get_total`).
5. **Delete with the button** (manager): "امسح مصروف البنزين". A card appears with the expense and "ينتهي هذا الطلب خلال
   10 دقائق"; press **تأكيد**: the card shows "تم الحذف ✅" and its buttons disappear. Asking the total again shows it
   without the 5000.
6. **A member asks about someone else** (the private window): "قديش صرف أبو أحمد هالشهر؟" with the manager's name. The
   agent declines politely with no number, and the step shows `forbidden` in red.
