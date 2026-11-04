"""Token measurement on 20 fixed messages (step 5a, §9.5). This script spends free-tier Groq quota.

    cd backend
    .venv\\Scripts\\python -m scripts.measure_tokens [--today 2026-11-04] [--phases small-low,main-low,small-default]

Each phase runs on a fresh copy of the seed database (never the development database) and an
in-process ChromaDB, with a fixed date. --messages picks messages; --setup runs messages first to
prepare the state (e.g. expenses to edit), marked "setup" so that they stay out of comparisons.
Results are saved after every message to backend/data/measurements/measure_<timestamp>.json
(untracked). Any 429 stops the whole run.
"""
import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import chromadb
import httpx
from sqlalchemy import select

from app.agent.context import ToolContext
from app.agent.loop import TurnOptions, run_turn
from app.db import seed
from app.db.models import Member
from app.db.session import init_db, make_engine, make_session_factory
from app.llm.client import GroqClient, LLMError
from app.rag.embeddings import OllamaEmbedder
from app.rag.vector_store import COLLECTIONS, ChromaVectorStore
from app.services.expenses import reindex_missing
from app.settings import BACKEND_DIR, get_settings
from app.tools.dispatch import dumps
from app.tools.schemas import TOOL_SCHEMAS

MESSAGES = [
    "دفعت 5000 بنزين",
    "اشتريت خضرا بـ 3000 مبارح",
    "جبت خضرا بـ3000 ولحمة بـ20000، قلي إذا تجاوزت ميزانية الأكل",
    "فاتورة الكهربا 15 ألف",
    "دفعت 50 لأبو أحمد",
    "كم صرفت هالشهر؟",
    "كم صرفنا على الأكل الشهر الماضي؟",
    "قارن مصاريف هالشهر مع الشهر الماضي",
    "وزعلي مصاريف هالشهر حسب الفئات",
    "كم باقي من ميزانية المواصلات؟",
    "حطلي ميزانية الترفيه 30 ألف بالشهر",
    "وين صرفت على الدكتور؟",
    "شو آخر 5 مصاريف سجلتها؟",
    "امسح آخر مصروف",
    "عدّل مصروف البنزين لـ 6000",
    "دفعت 20 دولار اشتراك نت",
    "كم صرف بابا هالشهر؟",
    "كيف بوفّر هالشهر؟",
    "حوّلي 100 ألف لأخي",
    "مرحبا كيفك",
]
MEMBER_MESSAGES = {17}  # sent as the member; all others as the manager
PHASES = {
    "small-low": ("small", "low", list(range(1, 21))),
    "main-low": ("main", "low", list(range(1, 21))),
    "small-default": ("small", None, [1, 3, 6, 8, 18]),
}


def _save(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _fresh_database(folder: Path, name: str, today: date):
    path = folder / f"{name}.db"
    engine = make_engine(f"sqlite:///{path.as_posix()}")
    init_db(engine)
    session = make_session_factory(engine)()
    seed.seed_categories(session)
    seed.seed_demo_family(session, "measure-only", "measure-only", today)  # throwaway database
    return engine, session


def _fresh_vector_store(embedder: OllamaEmbedder, session) -> tuple[ChromaVectorStore | None, int]:
    if not embedder.status()["ok"]:
        return None, 0
    client = chromadb.EphemeralClient()
    for name in COLLECTIONS:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    store = ChromaVectorStore("", 0, client=client)
    store.ensure_collections()
    return store, reindex_missing(session, embedder, store)


def _schema_size(client: GroqClient, model: str) -> dict:
    """Two tiny requests, with and without the tool schemas: the difference is what the schemas cost.

    (The real system prompt cannot be sent without tools: it mentions them, so the model tries to call one.)
    """
    messages = [{"role": "system", "content": "Reply with OK."}, {"role": "user", "content": "hi"}]
    try:
        bare = client.chat(model, messages, None).usage.prompt_tokens
        with_tools = client.chat(model, messages, TOOL_SCHEMAS).usage.prompt_tokens
    except LLMError as exc:
        return {"error": str(exc)}
    return {"bare_request": bare, "with_tools": with_tools, "tool_schemas": with_tools - bare}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--today", type=date.fromisoformat, default=date(2026, 11, 4))
    parser.add_argument("--phases", default=",".join(PHASES))
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between messages (the limiter adds more)")
    parser.add_argument("--messages", help="comma-separated message numbers (default: the phase's own list)")
    parser.add_argument("--setup", default="", help="message numbers run first in each phase, marked as setup")
    args = parser.parse_args(argv)
    phases = args.phases.split(",")
    chosen = [int(n) for n in args.messages.split(",")] if args.messages else None
    setup = [int(n) for n in args.setup.split(",") if n]

    settings = get_settings()
    folder = BACKEND_DIR / "data" / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = folder / f"measure_{stamp}.json"
    client = GroqClient.from_settings(settings, max_rate_limit_retries=0)
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    embedder.warmup()
    report = {"today": args.today.isoformat(), "started": stamp, "tool_mode": settings.tool_mode,
              "max_tokens": settings.llm_max_tokens, "setup": setup, "phases": {}, "rate_limits": {}}

    try:
        report["prompt_size"] = _schema_size(client, settings.model_small)
        print(f"prompt size: {report['prompt_size']}", flush=True)
        for phase in phases:
            model_name, effort, numbers = PHASES[phase]
            numbers = [(n, True) for n in setup] + [(n, False) for n in (chosen or numbers)]
            model = settings.model_id(model_name)
            client.reasoning_effort = effort
            engine, session = _fresh_database(folder, f"measure_{stamp}_{phase}", args.today)
            store, indexed = _fresh_vector_store(embedder, session)
            members = {m.role: m for m in session.scalars(select(Member))}
            results = []
            report["phases"][phase] = {"model": model, "reasoning_effort": effort or "default",
                                       "indexed_expenses": indexed, "messages": results}
            print(f"== {phase}: {model}, reasoning {effort or 'default'}, {indexed} expenses indexed", flush=True)

            for number, is_setup in numbers:
                text = MESSAGES[number - 1]
                who = members["member" if number in MEMBER_MESSAGES else "manager"]
                ctx = ToolContext(who.family_id, who.id, who.role, args.today, model)
                started = time.perf_counter()
                turn = run_turn(session, ctx, text, client, model=model, embedder=embedder if store else None,
                                vector_store=store, max_steps=settings.llm_max_steps,
                                options=TurnOptions(memory_messages=0))  # each message stands alone
                results.append({
                    "n": number, "setup": is_setup, "user": who.username, "text": text, "reply": turn.reply, "ended_by": turn.ended_by,
                    "llm_steps": turn.llm_steps,
                    "tools": [{"tool": s.tool, "arguments": s.arguments, "status": s.status,
                               "error": s.result.get("error")} for s in turn.steps],
                    "requests": turn.llm_calls, "tokens": turn.usage.as_dict(),
                    "seconds": round(time.perf_counter() - started, 1), "ungrounded": turn.ungrounded,
                    "pending_action": json.loads(dumps(turn.pending_action)), "error": turn.error,
                })
                report["rate_limits"] = client.last_rate_limits
                _save(out, report)
                tools = ", ".join(f"{s.tool}:{s.status}" for s in turn.steps) or "-"
                waited = sum(c.get("waited_ms", 0) for c in turn.llm_calls) / 1000
                print(f"  #{number:>2}{' (setup)' if is_setup else ''} [{turn.ended_by}] waited {waited:.1f}s, {turn.usage.total_tokens:>5} tok, "
                      f"{turn.llm_steps} step(s): {tools}", flush=True)
                if turn.error and "rate_limited" in turn.error:
                    print(f"STOPPED on a 429: {turn.error}", flush=True)
                    return 2
                time.sleep(args.pause)
            session.close()
            engine.dispose()
    except (LLMError, httpx.HTTPError) as exc:
        report["stopped"] = str(exc)
        _save(out, report)
        print(f"STOPPED: {exc}", flush=True)
        return 2
    finally:
        report["rate_limits"] = client.last_rate_limits
        _save(out, report)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
