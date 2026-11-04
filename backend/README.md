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
   Check: http://127.0.0.1:8000/health, Swagger: http://127.0.0.1:8000/docs

## Demo data

```
.venv\Scripts\python -m app.db.seed
```

Needs `SEED_MANAGER_PASSWORD` and `SEED_MEMBER_PASSWORD` in `backend\.env`. Creates the 12 categories and
"عائلة السالم" (base currency SYP_OLD, income 500,000) with users `manager` and `member` and 60 expenses.

## Tests

From the repository root (`pytest.ini` restricts collection to `backend/tests`):

```
backend\.venv\Scripts\python -m pytest
backend\.venv\Scripts\python -m pytest -m integration
```

The second command needs ChromaDB and Ollama running.
