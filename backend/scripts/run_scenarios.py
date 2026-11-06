"""Multi-turn development scenarios on the real models (step 6, §9.5, §13). This script spends free-tier Groq quota.

    cd backend
    .venv\\Scripts\\python -m scripts.run_scenarios [--today 2026-11-04] [--models main,small] [--only S1,S3]
    .venv\\Scripts\\python -m scripts.run_scenarios --resume data\\measurements\\scenarios_<timestamp>.json

Each scenario gets a new seed database and an in-process ChromaDB (never the development data), so that memory and
expenses cannot leak from one scenario into the next. Setup expenses go in through the services, without the model.
Results are saved after every turn to backend/data/measurements/scenarios_<timestamp>.json (untracked), and a scenario
is marked `done` as soon as it ends. The `checks` of each turn come from `expect` in dev_scenarios.json; `judgement` is
left for the person reading the results.

A short 429 is retried by the client. A retry-after over 60 s (the daily quota), or retries that fail, stop the run
cleanly: `stopped` in the file says why and where, with the error text and all the headers. --resume continues the same
file from the first unfinished scenario, on a new database; its finished turns move to `abandoned`, so that the tokens
they cost stay counted. These scenarios are not evaluation data.
"""
import argparse
import json
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import chromadb
from sqlalchemy import func, select

from app.agent.context import ToolContext
from app.agent.grounding import extract_numbers
from app.agent.loop import TurnOptions, TurnResult, run_turn
from app.agent.memory import memory_for_model, recent_messages
from app.db import seed
from app.db.models import Category, Expense, Member
from app.db.session import init_db, make_engine, make_session_factory
from app.llm.client import MAX_RETRY_WAIT, GroqClient
from app.rag.embeddings import OllamaEmbedder
from app.rag.vector_store import COLLECTIONS, ChromaVectorStore
from app.services.expenses import add_expense, reindex_missing
from app.services.periods import resolve_period
from app.settings import BACKEND_DIR, Settings, get_settings
from app.tools.dispatch import dumps

SCENARIOS = Path(__file__).with_name("dev_scenarios.json")


class CachedEmbedder:
    """The same 60 seed expenses are embedded for every scenario; Ollama only needs to see them once."""

    def __init__(self, embedder: OllamaEmbedder) -> None:
        self.embedder = embedder
        self.vectors: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text not in self.vectors:
            self.vectors[text] = self.embedder.embed(text)
        return self.vectors[text]


class RateLimited(Exception):
    """The client gave up on a 429: the run ends here."""

    def __init__(self, details: dict) -> None:
        super().__init__(f"{details['reason']} in {details['scenario']}.{details['turn']}")
        self.details = details


def _stop_details(scenario: dict, number: int, call: dict) -> dict:
    daily = call["retry_after"] is not None and call["retry_after"] > MAX_RETRY_WAIT
    return {"reason": "daily_quota" if daily else "retries_failed", "scenario": scenario["id"], "turn": number,
            "retry_after": call["retry_after"], "error": call["error"]}


def _vector_store() -> ChromaVectorStore:
    client = chromadb.EphemeralClient()
    for name in COLLECTIONS:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    store = ChromaVectorStore("", 0, client=client)
    store.ensure_collections()
    return store


def _matches(expected: Any, actual: Any, refs: dict[str, int]) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(_matches(v, actual.get(k), refs) for k, v in expected.items())
    if isinstance(expected, list):
        return any(_matches(option, actual, refs) for option in expected)
    if isinstance(expected, str) and expected.startswith("$"):
        return actual == refs[expected[1:]]
    return actual == expected


def _check_expense(session: Any, wanted: dict, refs: dict[str, int]) -> bool:
    session.expire_all()
    row = session.get(Expense, refs[wanted["ref"]])
    if (row is not None) != wanted["exists"]:
        return False
    return "amount" not in wanted or float(row.amount) == wanted["amount"]


def _used(steps: list, wanted: dict, refs: dict[str, int]) -> bool:
    return any(s.tool == wanted["tool"] and s.status == wanted["status"] and _matches(wanted["args"], s.arguments, refs)
               for s in steps)


def _reference_total(session: Any, ctx: ToolContext, wanted: dict) -> Decimal:
    """The answer computed here in SQL, not through the tools, so that they are not tested against themselves."""
    span = resolve_period(ctx, wanted["period"])
    query = select(func.sum(Expense.amount_base)).where(
        Expense.family_id == ctx.family_id, Expense.expense_date.between(span.start, span.end))
    if "category" in wanted:
        query = query.join(Category).where(Category.name_ar == wanted["category"])
    return Decimal(str(session.scalar(query) or 0)).normalize()


def _in_reply(value: Decimal, reply: str) -> bool:
    return value in {number.normalize() for number in extract_numbers(reply)}


def evaluate(expect: dict, turn: TurnResult, session: Any, refs: dict[str, int], ctx: ToolContext) -> list[dict]:
    """What the turn was expected to do, each as a separate check. Behaviour and result are judged, not the name of
    one tool: `achieved_by` lists every tool call that serves the purpose, and `reply_total` checks the answer."""
    steps = turn.steps
    checks = [{"check": "ended_by", "ok": _matches(expect["ended_by"], turn.ended_by, refs)}]
    for wanted in expect.get("tools", []):
        checks.append({"check": f"tool {wanted['tool']} {wanted['args']}", "ok": _used(steps, wanted, refs)})
    if "achieved_by" in expect:
        names = sorted({wanted["tool"] for wanted in expect["achieved_by"]})
        checks.append({"check": f"one of {names}", "ok": any(_used(steps, w, refs) for w in expect["achieved_by"])})
    if "clarifies" in expect:
        asked = [field for s in steps if s.tool == "ask_clarification" and s.status == "ok"
                 for field in s.arguments["missing_fields"]]
        wanted = expect["clarifies"]
        ok = all(f in asked for f in wanted.get("includes", [])) and not any(f in asked for f in wanted.get("excludes", []))
        checks.append({"check": f"asks about {wanted}", "ok": ok})
    if "reply_total" in expect:
        total = _reference_total(session, ctx, expect["reply_total"])
        checks.append({"check": f"reply has the reference {total:f}", "ok": _in_reply(total, turn.reply)})
    if "reply_number" in expect:
        checks.append({"check": f"reply has {expect['reply_number']}",
                       "ok": _in_reply(Decimal(expect["reply_number"]).normalize(), turn.reply)})
    if "preview_only_of" in expect and turn.pending_action:
        target = refs[expect["preview_only_of"][1:]]
        checks.append({"check": "a preview, if any, is of the right expense", "ok": turn.pending_action["expense"]["id"] == target})
    for name in expect.get("no_tools", []):
        checks.append({"check": f"no {name}", "ok": all(s.tool != name for s in steps)})
    if "llm_calls" in expect:
        checks.append({"check": f"{expect['llm_calls']} model calls", "ok": len(turn.llm_calls) == expect["llm_calls"]})
    if expect.get("no_model_free"):
        checks.append({"check": "not the model-free path", "ok": turn.llm_steps > 0})
    if "any_status" in expect:
        checks.append({"check": f"a tool got {expect['any_status']}", "ok": any(s.status == expect["any_status"] for s in steps)})
    for text in expect.get("reply_has", []):
        checks.append({"check": f"reply has {text}", "ok": text in turn.reply})
    for text in expect.get("reply_lacks", []):
        checks.append({"check": f"reply lacks {text}", "ok": text not in turn.reply})
    if "expense" in expect:
        checks.append({"check": f"expense {expect['expense']}", "ok": _check_expense(session, expect["expense"], refs)})
    if "source_text" in expect:
        latest = session.scalars(select(Expense).order_by(Expense.id.desc())).first()
        checks.append({"check": "source_text", "ok": latest.source_text == expect["source_text"]})
    if "search_finds" in expect:
        target = refs[expect["search_finds"][1:]]
        found = any(row["id"] == target for s in steps if s.tool == "search_expenses" and s.status == "ok"
                    for row in s.result["data"]["matches"])
        checks.append({"check": "search finds the expense", "ok": found})
    if expect.get("no_ungrounded"):
        checks.append({"check": "no ungrounded numbers in the final reply", "ok": not turn.ungrounded})
    return checks


def _turn_record(number: int, text: str, turn: TurnResult, memory: list[dict], seconds: float, checks: list[dict]) -> dict:
    return {
        "turn": number, "text": text, "reply": turn.reply, "ended_by": turn.ended_by,
        "tools": [{"tool": s.tool, "arguments": s.arguments, "status": s.status} for s in turn.steps],
        "model_calls": sum(1 for c in turn.llm_calls if c["status"] in ("ok", "empty", "truncated")),
        "tokens": turn.usage.as_dict(), "empty_retried": turn.empty_retried,
        "calls": [{"step": c["step"], "attempt": c["attempt"], "status": c["status"], "waited_ms": c["waited_ms"],
                   "latency_ms": c["latency_ms"], **c["tokens"],
                   **({"empty_details": json.loads(c["error"])} if c["status"] == "empty" else {})}
                  for c in turn.llm_calls],
        "rate_limit_events": [{"step": c["step"], "attempt": c["attempt"], "retry_after": c["retry_after"],
                               "error": c["error"]} for c in turn.llm_calls if c["status"] == "rate_limited"],
        "first_call_prompt_tokens": next((c["tokens"]["prompt"] for c in turn.llm_calls if c["tokens"]["prompt"]), None),
        "memory_messages": len(memory), "repaired": turn.repaired, "first_reply": turn.first_reply,
        "first_ungrounded": turn.first_ungrounded, "ungrounded": turn.ungrounded, "year_corrected": turn.year_corrected,
        "waited_s": round(sum(c.get("waited_ms", 0) for c in turn.llm_calls) / 1000, 1), "seconds": seconds,
        "pending_action": json.loads(dumps(turn.pending_action)), "error": turn.error, "checks": checks,
        "passed": all(c["ok"] for c in checks), "judgement": None,
    }


def run_scenario(scenario: dict, result: dict, settings: Settings, client: GroqClient, model: str,
                 embedder: CachedEmbedder, today: date, save: Any) -> None:
    """Fills `result`, which is already in the report, so that a stopped scenario keeps its finished turns."""
    result.update(who=scenario["who"], done=False, turns=[])
    options = TurnOptions.from_settings(settings)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        engine = make_engine(f"sqlite:///{(Path(folder) / 'scenario.db').as_posix()}")
        init_db(engine)
        with make_session_factory(engine)() as session:
            seed.seed_categories(session)
            seed.seed_demo_family(session, "dev-only", "dev-only", today)  # throwaway database
            store = _vector_store()
            result["indexed"] = reindex_missing(session, embedder, store)
            who = session.scalar(select(Member).where(Member.username == scenario["who"]))
            ctx = ToolContext(who.family_id, who.id, who.role, today, model)
            refs = {}
            for item in scenario["setup"]:
                refs[item["ref"]] = add_expense(
                    session, ctx, amount=item["amount"], category=item["category"], description=item["description"],
                    expense_date=today - timedelta(days=item["days_ago"]), embedder=embedder, vector_store=store)["id"]
            result["refs"] = refs

            for number, step in enumerate(scenario["turns"], start=1):
                memory = memory_for_model(recent_messages(session, ctx, options.memory_messages, options.memory_hours),
                                          options.memory_max_tokens)
                started = time.perf_counter()
                turn = run_turn(session, ctx, step["text"], client, model=model, embedder=embedder, vector_store=store,
                                max_steps=settings.llm_max_steps, options=options)
                seconds = round(time.perf_counter() - started, 1)
                checks = evaluate(step["expect"], turn, session, refs, ctx)
                result["turns"].append(_turn_record(number, step["text"], turn, memory, seconds, checks))
                save()
                record = result["turns"][-1]
                tools = ", ".join(f"{t['tool']}:{t['status']}" for t in record["tools"]) or "-"
                mark = "OK " if record["passed"] else "?? "
                print(f"  {scenario['id']}.{number} {mark}[{turn.ended_by}] {record['model_calls']} calls, "
                      f"{record['tokens']['total']} tok, repair {turn.repaired}: {tools}", flush=True)
                if turn.llm_calls and turn.llm_calls[-1]["status"] == "rate_limited":
                    raise RateLimited(_stop_details(scenario, number, turn.llm_calls[-1]))
        engine.dispose()
    result["done"] = True
    save()


def _summary(run: dict) -> dict:
    turns = [t for s in run["scenarios"] for t in s["turns"]]
    abandoned = [t for s in run.get("abandoned", []) for t in s["turns"]]
    calls = [c for t in turns + abandoned for c in t["calls"]]
    first = [t["first_call_prompt_tokens"] for t in turns if t["first_call_prompt_tokens"] and not t["memory_messages"]]
    later = [t["first_call_prompt_tokens"] for t in turns if t["first_call_prompt_tokens"] and t["memory_messages"]]
    return {
        "scenarios_done": sum(1 for s in run["scenarios"] if s["done"]),
        "turns": len(turns), "model_calls": sum(t["model_calls"] for t in turns),
        "tokens": sum(t["tokens"]["total"] for t in turns),
        "tokens_in_abandoned_turns": sum(t["tokens"]["total"] for t in abandoned),
        "waited_s": round(sum(c["waited_ms"] for c in calls) / 1000, 1),
        "rate_limited_attempts": sum(1 for c in calls if c["status"] == "rate_limited"),
        "truncated_replies": sum(1 for c in calls if c["status"] == "truncated"),
        "empty_replies": sum(1 for c in calls if c["status"] == "empty"),
        "largest_completion": max((c["completion"] for c in calls), default=0),
        "prompt_tokens_first_call_without_memory": round(sum(first) / len(first)) if first else None,
        "prompt_tokens_first_call_with_memory": round(sum(later) / len(later)) if later else None,
        "turns_with_memory": len(later), "turns_without_the_model": sum(1 for t in turns if t["model_calls"] == 0),
        "repairs": sum(1 for t in turns if t["repaired"]),
        "repairs_cleared": sum(1 for t in turns if t["repaired"] and not t["ungrounded"]),
        "checks_passed": sum(1 for t in turns if t["passed"]),
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--today", type=date.fromisoformat, default=date(2026, 11, 4))
    parser.add_argument("--models", default="main,small")
    parser.add_argument("--only", help="comma-separated scenario ids")
    parser.add_argument("--resume", type=Path, help="a results file to continue from its first unfinished scenario")
    args = parser.parse_args(argv)

    if args.resume:
        path = args.resume
        report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if "stopped" in report:
            report.setdefault("earlier_stops", []).append(report.pop("stopped"))
    else:
        folder = BACKEND_DIR / "data" / "measurements"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = folder / f"scenarios_{stamp}.json"
        report = {"started": stamp, "today": args.today.isoformat(), "reasoning_effort": "low",
                  "models": args.models.split(","), "only": args.only.split(",") if args.only else None, "runs": {}}
    today = date.fromisoformat(report["today"])
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]
    if report["only"]:
        scenarios = [s for s in scenarios if s["id"] in report["only"]]
    settings = get_settings().model_copy(update={"agent_today": today, "reasoning_effort": "low"})

    def save() -> None:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    client = GroqClient.from_settings(settings)
    ollama = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    report["ollama_warmup"] = ollama.warmup()
    if not report["ollama_warmup"]["ok"]:
        print("Stopped before any model call: Ollama is not ready.", file=sys.stderr)
        return 1
    embedder = CachedEmbedder(ollama)

    code, current = 0, ""
    try:
        for name in report["models"]:
            model = settings.model_id(name)
            run = report["runs"].setdefault(name, {"model": model, "scenarios": []})
            unfinished = [s for s in run["scenarios"] if not s["done"]]
            run["scenarios"] = [s for s in run["scenarios"] if s["done"]]
            if unfinished:
                run.setdefault("abandoned", []).extend(unfinished)
            finished = {s["id"] for s in run["scenarios"]}
            current = name
            print(f"== {name}: {model}, {len(finished)} of {len(scenarios)} already done", flush=True)
            for scenario in scenarios:
                if scenario["id"] in finished:
                    continue
                result = {"id": scenario["id"]}
                run["scenarios"].append(result)
                run_scenario(scenario, result, settings, client, model, embedder, today, save)
                run["summary"] = _summary(run)
                run["rate_limits"] = client.last_rate_limits.get(model)
                save()
    except RateLimited as exc:
        report["stopped"] = {**exc.details, "model": current, "rate_limits": client.last_rate_limits.get(
            settings.model_id(current))}
        print(f"STOPPED ({exc}) on {current}: retry-after {exc.details['retry_after']}. Nothing is retried now.", flush=True)
        print(f"To continue: python -m scripts.run_scenarios --resume {path}", flush=True)
        code = 2
    finally:
        for run in report["runs"].values():
            run["summary"] = _summary(run)
            run["rate_limits"] = client.last_rate_limits.get(run["model"]) or run.get("rate_limits")
        save()
        client.close()
    print(f"saved {path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
