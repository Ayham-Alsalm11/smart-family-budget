"""One real run on Groq through the API itself (step 5b, §13). This script spends free-tier Groq quota.

    cd backend
    .venv\\Scripts\\python -m scripts.api_smoke [--today 2026-11-04]

One family on a new temporary database and an in-process ChromaDB (never the development data),
with a fixed date, gpt-oss-120b and reasoning_effort=low:

1. The scenario: register, invite, join, then chat and direct API calls in order. Each step states what is expected.
2. The prompt-cache question: the same full request 3 times. The rate-limit bucket is left to refill completely
   before each request, so that remaining-before is the limit and remaining-after shows what the request was charged.
   Its date differs from the scenario's, so that the first request cannot find the prompt in the cache.

Results go to backend/data/measurements/api_smoke_<timestamp>.json (untracked), after every step. A short 429 is
retried by the client; a retry-after over 60 s, or retries that fail, stop the run. The API key, passwords and tokens
are never printed or saved.
"""
import argparse
import json
import secrets
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import chromadb
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.agent.prompts import build_system_prompt
from app.db.models import LlmCall, ToolLog
from app.llm.client import GroqClient, LLMError
from app.main import create_app
from app.rag.embeddings import OllamaEmbedder
from app.rag.vector_store import ChromaVectorStore
from app.settings import BACKEND_DIR, get_settings
from app.tools.schemas import TOOL_SCHEMAS

MANAGER_NAME = "أبو أحمد"
MEMBER_NAME = "أم أحمد"
CACHE_MESSAGE = "مرحبا"
CACHE_REPEATS = 3
EIGHT_THOUSAND = ("8000", "8,000", "8٬000", "٨٠٠٠", "8 آلاف", "8 الاف", "٨ آلاف")


class Stop(Exception):
    """A 429 the client gave up on: the run ends here."""


class Run:
    def __init__(self, api: TestClient, path: Path, report: dict) -> None:
        self.api = api
        self.path = path
        self.report = report

    def save(self) -> None:
        self.path.write_text(json.dumps(self.report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def record(self, step: dict) -> dict:
        self.report["steps"].append(step)
        self.save()
        mark = "OK " if step.get("as_expected") else "NOT AS EXPECTED"
        print(f"[{step['n']}] {mark} {step['title']}: {step.get('summary', '')}", flush=True)
        return step

    def _logs(self, message_id: int) -> tuple[list[dict], list[dict], list[str]]:
        with self.api.app.state.session_factory() as session:
            calls = session.scalars(select(LlmCall).where(LlmCall.message_id == message_id).order_by(LlmCall.id)).all()
            tools = session.scalars(select(ToolLog).where(ToolLog.message_id == message_id).order_by(ToolLog.id)).all()
        llm = [{"step": c.step, "attempt": c.attempt, "status": c.status, "http_status": c.http_status,
                "latency_ms": c.latency_ms, "prompt": c.prompt_tokens, "completion": c.completion_tokens,
                "reasoning": c.reasoning_tokens, "cached": c.cached_tokens} for c in calls]
        results = [{"tool": t.tool_name, "arguments": t.arguments, "status": t.status, "result": t.result}
                   for t in tools if t.status != "ungrounded"]
        ungrounded = [n for t in tools if t.status == "ungrounded" for n in t.arguments["numbers"]]
        return llm, results, ungrounded

    def chat(self, n: int, who: str, headers: dict, text: str) -> dict:
        started = time.perf_counter()
        response = self.api.post("/chat", json={"message": text}, headers=headers)
        seconds = round(time.perf_counter() - started, 1)
        if response.status_code != 200:
            return {"n": n, "who": who, "text": text, "http_status": response.status_code, "reply": None, "body": response.json(),
                    "tools": [], "llm_calls": [], "tokens": {}, "seconds": seconds}
        body = response.json()
        llm, tools, ungrounded = self._logs(body["message_id"])
        tokens = {key: sum(call[key] or 0 for call in llm) for key in ("prompt", "completion", "reasoning", "cached")}
        step = {"n": n, "who": who, "text": text, "http_status": 200, "reply": body["reply"], "tools": tools,
                "pending_action": body.get("pending_action"), "alerts": body["alerts"], "llm_calls": llm,
                "tokens": {**tokens, "total": tokens["prompt"] + tokens["completion"]}, "ungrounded": ungrounded,
                "seconds": seconds}
        if llm and llm[-1]["status"] == "rate_limited":
            step["title"], step["as_expected"] = text, False
            self.record(step)
            raise Stop(f"429 in step {n}")
        return step


def _tool_statuses(step: dict) -> str:
    return ", ".join(f"{t['tool']}:{t['status']}" for t in step["tools"]) or "no tools"


def _added(step: dict) -> dict | None:
    for tool in step["tools"]:
        if tool["tool"] == "add_expense" and tool["status"] == "ok":
            return tool["result"]["data"]["expense"]
    return None


def scenario(run: Run, password: str) -> None:
    api = run.api
    response = api.post("/auth/register-family", json={
        "family_name": "عائلة التجربة", "monthly_income": 500000, "base_currency": "SYP_OLD", "name": MANAGER_NAME,
        "username": "smoke_manager", "password": password})
    manager = {"Authorization": f"Bearer {response.json()['access_token']}"}
    run.record({"n": 1, "title": "manager registers a family in SYP_OLD", "http_status": response.status_code,
                "as_expected": response.status_code == 201, "summary": f"HTTP {response.status_code}"})

    code = api.post("/invites", json={"role": "member"}, headers=manager).json()["code"]
    response = api.post("/auth/join", json={"code": code, "name": MEMBER_NAME, "username": "smoke_member",
                                            "password": password})
    member = {"Authorization": f"Bearer {response.json()['access_token']}"}
    role = response.json()["member"]["role"]
    run.record({"n": 2, "title": "member joins with an invite", "http_status": response.status_code, "role": role,
                "as_expected": response.status_code == 201 and role == "member", "summary": f"role {role}"})

    step = run.chat(3, "manager", manager, "دفعت 5000 بنزين")
    added = _added(step)
    step["title"] = "manager records fuel"
    step["as_expected"] = bool(added) and added["amount"] == 5000 and added["category"] == "بنزين ومواصلات"
    step["summary"] = f"{_tool_statuses(step)} | {step['reply']}"
    run.record(step)

    step = run.chat(4, "member", member, "اشتريت خضرا بـ 3000")
    added = _added(step)
    step["title"] = "member records vegetables"
    step["as_expected"] = bool(added) and added["amount"] == 3000 and added["category"] == "أكل وشرب"
    step["summary"] = f"{_tool_statuses(step)} | {step['reply']}"
    run.record(step)

    step = run.chat(5, "member", member, f"كم صرف {MANAGER_NAME} هالشهر؟")
    step["title"] = "member asks about the manager"
    step["as_expected"] = any(t["status"] == "forbidden" for t in step["tools"]) and "5000" not in (step["reply"] or "")
    step["summary"] = f"{_tool_statuses(step)} | {step['reply']}"
    run.record(step)

    step = run.chat(6, "manager", manager, "كم صرفنا هالشهر؟")
    step["title"] = "manager asks for the family total"
    step["as_expected"] = any(text in (step["reply"] or "") for text in EIGHT_THOUSAND)
    step["summary"] = f"{_tool_statuses(step)} | {step['reply']}"
    run.record(step)

    rows = api.get("/expenses", headers=manager).json()["expenses"]
    fuel = next(row for row in rows if row["category"] == "بنزين ومواصلات")
    response = api.delete(f"/expenses/{fuel['id']}", headers=member)
    run.record({"n": 7, "title": "member deletes the manager's expense through the API",
                "http_status": response.status_code, "error": response.json().get("error"),
                "as_expected": response.status_code == 403, "summary": f"HTTP {response.status_code}"})

    step = run.chat(8, "manager", manager, "امسح مصروف البنزين")
    step["title"] = "manager deletes the fuel expense, then confirms"
    pending = step.get("pending_action")
    deleted = False
    if pending:
        before = [row["id"] for row in api.get("/expenses", headers=manager).json()["expenses"]]
        confirm = api.post(f"/actions/{pending['confirmation_id']}/confirm", headers=manager)
        after = [row["id"] for row in api.get("/expenses", headers=manager).json()["expenses"]]
        deleted = confirm.status_code == 200 and fuel["id"] in before and fuel["id"] not in after
        step["confirm_http_status"] = confirm.status_code
        step["total_after"] = api.get("/summary", headers=manager).json()["total"]
    step["as_expected"] = deleted and pending["expense"]["id"] == fuel["id"]
    step["summary"] = f"{_tool_statuses(step)} | {step['reply']} | deleted after confirm: {deleted}"
    run.record(step)


def _remaining(client: GroqClient, model: str) -> tuple[int, int]:
    headers = client.last_rate_limits[model]
    return int(headers["x-ratelimit-remaining-tokens"]), int(headers["x-ratelimit-limit-tokens"])


def _wait_for_full_bucket(client: GroqClient, model: str, limit_per_minute: int) -> float:
    remaining, limit = _remaining(client, model)
    seconds = (limit - remaining) * 60 / limit_per_minute + 1
    time.sleep(seconds)
    return round(seconds, 1)


def cache_probe(run: Run, client: GroqClient, model: str, today: date) -> None:
    """The same request CACHE_REPEATS times. charged = remaining before + refill during the call - remaining after."""
    messages = [{"role": "system", "content": build_system_prompt(today + timedelta(days=1), "SYP_OLD")},
                {"role": "user", "content": CACHE_MESSAGE}]
    probe = {"model": model, "message": CACHE_MESSAGE, "calls": []}
    run.report["cache_probe"] = probe
    for n in range(1, CACHE_REPEATS + 1):
        _, limit = _remaining(client, model)
        waited = _wait_for_full_bucket(client, model, limit)
        started = time.perf_counter()
        response = client.chat(model, messages, TOOL_SCHEMAS)
        seconds = time.perf_counter() - started
        after, _ = _remaining(client, model)
        usage = response.usage
        charged = limit + round(seconds * limit / 60) - after
        call = {"n": n, "waited_for_refill_s": waited, "seconds": round(seconds, 2), "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens, "cached_tokens": usage.cached_tokens,
                "total_tokens": usage.total_tokens, "remaining_before": limit, "remaining_after": after,
                "charged": charged, "uncached_total": usage.total_tokens - usage.cached_tokens}
        probe["calls"].append(call)
        run.save()
        print(f"cache call {n}: {call}", flush=True)

    cached = [c for c in probe["calls"] if c["cached_tokens"] > 0]
    probe["cached_tokens_seen"] = bool(cached)
    probe["cached_tokens_free_of_the_minute_limit"] = None if not cached else all(
        abs(c["charged"] - c["uncached_total"]) < abs(c["charged"] - c["total_tokens"]) for c in cached)
    run.save()


def _vector_store() -> ChromaVectorStore:
    client = chromadb.EphemeralClient()
    for collection in client.list_collections():
        client.delete_collection(collection.name)
    store = ChromaVectorStore("", 0, client=client)
    store.ensure_collections()
    return store


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--today", type=date.fromisoformat, default=date(2026, 11, 4))
    args = parser.parse_args(argv)

    folder = BACKEND_DIR / "data" / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"api_smoke_{stamp}.json"
    report: dict[str, Any] = {"started": stamp, "today": args.today.isoformat(), "steps": []}

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        settings = get_settings().model_copy(update={
            "database_url": f"sqlite:///{(Path(directory) / 'api_smoke.db').as_posix()}", "agent_today": args.today,
            "agent_model": "main", "reasoning_effort": "low"})
        model = settings.model_id("main")
        report.update(model=model, reasoning_effort=settings.reasoning_effort)
        client = GroqClient.from_settings(settings)
        embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
        report["ollama_warmup"] = embedder.warmup()
        store = _vector_store()
        app = create_app(settings, embedder=embedder, vector_store_factory=lambda _settings: store, llm_client=client)
        code = 0
        with TestClient(app) as api:
            run = Run(api, path, report)
            try:
                scenario(run, secrets.token_urlsafe(16))
                run.save()
                cache_probe(run, client, model, args.today)
            except Stop as exc:
                report["stopped"] = str(exc)
                print(f"STOPPED: {exc}", flush=True)
                code = 2
            except LLMError as exc:
                report["stopped"] = f"{exc.kind}: {exc}"
                print(f"STOPPED: {report['stopped']}", flush=True)
                code = 2
            finally:
                report["rate_limits"] = client.last_rate_limits.get(model)
                turns = [s["tokens"]["total"] for s in report["steps"] if s.get("tokens")]
                report["scenario_tokens"] = sum(turns)
                run.save()
        client.close()

    if code == 0 and not all(step.get("as_expected") for step in report["steps"]):
        code = 1
    print(f"saved {path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
