# Backend (agent-api)

FastAPI service of the Family Budget Agent. Specification: [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Setup (Windows, once)

```
cd backend
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

Fill in `backend\.env` yourself (`GROQ_API_KEY`, `JWT_SECRET`, seed passwords). Never commit or print it.
`JWT_SECRET` needs at least 32 characters, otherwise the login paths answer 503. To generate one:
`.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(48))"`

## Running the three local services

Each in its own terminal, from `backend\`:

1. **ChromaDB server** (port 8001, data in `backend\data\chroma`):
   ```
   .venv\Scripts\chroma run --path data\chroma --host localhost --port 8001
   ```
2. **Ollama** with BGE-M3 (port 11434). If Ollama is not already running in the tray:
   ```
   ollama serve
   ```
   Once: `ollama pull bge-m3`. The first embedding after a cold start takes 10–30 s; the API warms it up at start-up.
3. **API** (port 8000; creates `backend\data\fba.db` on first start):
   ```
   .venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   Check: http://127.0.0.1:8000/health, Swagger: http://127.0.0.1:8000/docs (use **Authorize** with the token).
   Without ChromaDB or Ollama the API still runs and search by meaning answers `not_available`. Without
   `GROQ_API_KEY`, `/chat` answers 503.

## Trying the API with curl (Windows `cmd`)

Register a family; the reply has an `access_token`:

```
curl -X POST http://127.0.0.1:8000/auth/register-family -H "Content-Type: application/json" -d "{\"family_name\":\"Al Salem\",\"monthly_income\":500000,\"base_currency\":\"SYP_OLD\",\"name\":\"Abu Ahmad\",\"username\":\"abu.ahmad\",\"password\":\"choose-a-long-one\"}"
set TOKEN=paste-the-access-token-here
```

Send a message to the agent. This one spends Groq quota. `cmd` garbles Arabic on the command line, so put the
body in a UTF-8 file `chat.json` containing `{"message": "دفعت 5000 بنزين"}`:

```
curl -X POST http://127.0.0.1:8000/chat -H "Authorization: Bearer %TOKEN%" -H "Content-Type: application/json" --data-binary @chat.json
```

Every error has the same shape: `{"error": {"code": "...", "message": "...", "fields": ["..."]}}`, and `fields` is
present only when it applies.

## Demo data

```
.venv\Scripts\python -m app.db.seed
```

Needs `SEED_MANAGER_PASSWORD` and `SEED_MEMBER_PASSWORD` in `backend\.env`. Creates the 12 categories and
"عائلة السالم" (base currency SYP_OLD, income 500,000) with users `manager` and `member` and 60 expenses.

Search by meaning needs these expenses in ChromaDB. **After every re-creation of the development database**, with
ChromaDB and Ollama running:

```
.venv\Scripts\python -m app.rag.reindex --reset
```

It wipes the server's collections and embeds every expense of the current database again. The server keeps its data
apart from `fba.db`, so without this step search finds nothing, or old vectors stay under ids that now mean other
expenses. Without `--reset` it only embeds the expenses that have no vector yet.

## Tests

From the repository root (`pytest.ini` restricts collection to `backend/tests` and `eval/tests`):

```
backend\.venv\Scripts\python -m pytest
backend\.venv\Scripts\python -m pytest -m integration
```

The second command needs ChromaDB and Ollama running.
