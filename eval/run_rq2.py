"""The RQ2 runner (§9.4): the same questions on the agent and on GP1, on the same records. Spends free-tier Groq quota.

    backend\\.venv\\Scripts\\python -m eval.run_rq2 --questions data\\eval\\pilot_numeric_questions.jsonl
    backend\\.venv\\Scripts\\python -m eval.run_rq2 --questions data\\eval\\pilot_numeric_questions.jsonl --resume <run_id>

For each question the agent answers first: a new member each time, memory off, corrections off, AGENT_TODAY = the
database's date, gpt-oss-120b at `low`. GP1 then answers through eval/gp1_answer.py. GP1 uses the real date, so the run is
refused unless the database was generated today (§9.2). A run stopped by the daily quota continues with --resume after a
new generation: each item keeps the date and the records hash it ran on, and its own reference.

Writes results/<run_id>/: config.json, items.jsonl (one line as soon as an item ends), summary.json and summary.md.
Each run works on its own copy of the databases, under data/eval/generated/runs/, so that the generated ones stay clean.
A failed request counts as wrong (§9.3).
"""
import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from eval import DATA_DIR, EVAL_DIR, GENERATED, GP1_PYTHON, RESULTS, ROOT
from eval.questions import TYPES, load_questions, reference
from eval.scoring import score

import chromadb  # noqa: E402
from sqlalchemy import delete, func, select  # noqa: E402

from app.agent.context import ToolContext  # noqa: E402
from app.agent.loop import TurnOptions, run_turn  # noqa: E402
from app.db.models import Budget, Expense, Family, Member  # noqa: E402
from app.db.session import make_engine, make_session_factory  # noqa: E402
from app.llm.client import MAX_RETRY_WAIT, GroqClient  # noqa: E402
from app.rag.embeddings import OllamaEmbedder  # noqa: E402
from app.rag.vector_store import ChromaVectorStore  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402

SYSTEMS = ("agent", "gp1")
EFFORT = "low"
GP1_MAX_TOKENS = {"parser": 200, "engine": 500}  # decision A: GP1's limits stay as they are
GP1_TOKENS_RESERVED = 3500  # two GP1 requests with their max_tokens; the agent's limiter must know they are coming
DAILY_QUOTA_EXIT = 2


class DailyQuota(Exception):
    """The run ends here, and the item that hit the limit stays unfinished."""


class WrongDay(Exception):
    pass


def read_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_database(folder: Path, today: date) -> tuple[dict[str, Any], dict[str, Any]]:
    """The manifest and the records, or a clear refusal."""
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise WrongDay(f"no generated database in {folder}: run python -m eval.generate_eval_db")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["today"] != today.isoformat():
        raise WrongDay(f"the database is dated {manifest['today']} and today is {today}. GP1 uses the real date, so the "
                       "experiment must run on the day of the database: run python -m eval.generate_eval_db --overwrite")
    data = (folder / "records.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["records_sha256"]:
        raise WrongDay("records.json does not match the manifest: generate the database again")
    return manifest, json.loads(data)


def build_item(question: dict[str, Any], system: str, wanted: dict[str, Any], answer: dict[str, Any],
               manifest: dict[str, Any]) -> dict[str, Any]:
    """`answer` is what a system returned; scoring is the same for both, and a failure is wrong whatever the text says."""
    scored = score(answer["reply"], wanted["value"])
    failed = answer.get("error") is not None
    return {
        "id": question["id"], "type": question["type"], "question": question["question"], "system": system,
        "reference": wanted["value"], "reference_detail": wanted.get("detail"), "reply": answer["reply"],
        "numbers": scored["numbers"], "has_number": scored["has_number"],
        "strict": scored["strict"] and not failed, "lenient": scored["lenient"] and not failed,
        "error": answer.get("error"), "tokens": answer["tokens"], "requests": answer["requests"],
        "seconds": answer["seconds"], "waited_s": answer["waited_s"], "db_date": manifest["today"],
        "records_sha256": manifest["records_sha256"], "details": answer.get("details", {}),
    }


def run_items(questions: list[dict[str, Any]], records: dict[str, Any], manifest: dict[str, Any],
              systems: dict[str, Callable[[dict[str, Any]], dict[str, Any]]], items_path: Path) -> int:
    """Appends every missing (question, system) pair. Returns how many were added; DailyQuota passes through."""
    done = {(item["id"], item["system"]) for item in read_items(items_path)}
    added = 0
    for question in questions:
        wanted = reference(question, records)
        for system in SYSTEMS:
            if (question["id"], system) in done:
                continue
            item = build_item(question, system, wanted, systems[system](question), manifest)
            with items_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
            added += 1
            print(f"{question['id']} {system}: strict {item['strict']}, lenient {item['lenient']}, "
                  f"{item['tokens']['total']} tok, {item['requests']} req, {item['seconds']} s"
                  + (f", error: {item['error'][:80]}" if item["error"] else ""), flush=True)
    return added


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _block(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    return {
        "items": count, "strict": sum(i["strict"] for i in items), "lenient": sum(i["lenient"] for i in items),
        "strict_accuracy": _share(sum(i["strict"] for i in items), count),
        "lenient_accuracy": _share(sum(i["lenient"] for i in items), count),
    }


def summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for system in SYSTEMS:
        chosen = [item for item in items if item["system"] == system]
        count = len(chosen)
        summary[system] = {
            **_block(chosen),
            "failures": sum(1 for i in chosen if i["error"]),
            "api_failure_rate": _share(sum(1 for i in chosen if i["error"]), count),
            "replies_without_a_number": sum(1 for i in chosen if not i["has_number"]),
            # Without the time spent waiting for the rate limit: that depends on the experiment, not on the system.
            "mean_seconds": round(mean(i["seconds"] - i["waited_s"] for i in chosen), 2) if chosen else None,
            "mean_waited_s": round(mean(i["waited_s"] for i in chosen), 2) if chosen else None,
            "requests_per_question": round(mean(i["requests"] for i in chosen), 2) if chosen else None,
            "tokens_per_question": round(mean(i["tokens"]["total"] for i in chosen)) if chosen else None,
            "tokens_total": sum(i["tokens"]["total"] for i in chosen),
            "by_type": {kind: _block([i for i in chosen if i["type"] == kind]) for kind in TYPES
                        if any(i["type"] == kind for i in chosen)},
        }
    return summary


def summary_markdown(config: dict[str, Any], summary: dict[str, Any]) -> str:
    rows = [("Strict accuracy", "strict_accuracy"), ("Lenient accuracy", "lenient_accuracy"), ("Failures", "failures"),
            ("API failure rate", "api_failure_rate"), ("Replies without a number", "replies_without_a_number"),
            ("Mean seconds, without rate-limit waits", "mean_seconds"), ("Mean rate-limit wait, s", "mean_waited_s"),
            ("Requests per question", "requests_per_question"), ("Tokens per question", "tokens_per_question"),
            ("Tokens in total", "tokens_total")]
    lines = [f"# RQ2 run {config['run_id']}", "",
             f"`{config['model']}` at `{config['reasoning_effort']}`, git `{config['git_commit'][:8]}`, questions "
             f"`{config['questions_file']}`, database of {config['segments'][-1]['today']}.", ""]
    if config.get("stopped"):
        lines += [f"**Stopped:** {config['stopped']}", ""]
    lines += ["| | Agent | GP1 |", "|---|---|---|"]
    lines += [f"| {title} | {summary['agent'][key]} | {summary['gp1'][key]} |" for title, key in rows]
    lines += ["", "| Type | Agent strict | Agent lenient | GP1 strict | GP1 lenient |", "|---|---|---|---|---|"]
    for kind in TYPES:
        cells = []
        for system in SYSTEMS:
            block = summary[system]["by_type"].get(kind)
            cells += [f"{block['strict']}/{block['items']}", f"{block['lenient']}/{block['items']}"] if block else ["-", "-"]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_summary(folder: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary = summarise(read_items(folder / "items.jsonl"))
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / "summary.md").write_text(summary_markdown(config, summary), encoding="utf-8")
    return summary


# ---------- the two systems ----------

class AgentSystem:
    def __init__(self, session: Any, settings: Settings, client: GroqClient, model: str, embedder: Any, store: Any,
                 today: date) -> None:
        self.session, self.settings, self.client, self.model = session, settings, client, model
        self.embedder, self.store, self.today = embedder, store, today
        self.family_id = session.scalar(select(Family.id))
        self.newest_id = session.scalar(select(func.max(Expense.id)))

    def _undo_writes(self) -> list[float]:
        """A question must leave the records as they were, or the next reference would not hold."""
        rows = self.session.scalars(select(Expense).where(Expense.id > self.newest_id)).all()
        added = [float(row.amount) for row in rows]
        for row in rows:
            self.session.delete(row)
            self.store.delete("expenses", str(row.id))
        self.session.execute(delete(Budget))
        self.session.commit()
        return added

    def __call__(self, question: dict[str, Any]) -> dict[str, Any]:
        # A new member, so that nothing of an earlier question can reach this one, even with memory off (§9.4).
        member = Member(family_id=self.family_id, name="مقيّم", username=f"eval_{secrets.token_hex(6)}", password_hash="!",
                        role="manager")
        self.session.add(member)
        self.session.commit()
        ctx = ToolContext(self.family_id, member.id, member.role, self.today, self.model)
        started = time.perf_counter()
        turn = run_turn(self.session, ctx, question["question"], self.client, model=self.model, embedder=self.embedder,
                        vector_store=self.store, max_steps=self.settings.llm_max_steps,
                        options=TurnOptions.from_settings(self.settings))
        seconds = time.perf_counter() - started
        last = turn.llm_calls[-1] if turn.llm_calls else None
        if last and last["status"] == "rate_limited" and (last["retry_after"] or 0) > MAX_RETRY_WAIT:
            self._undo_writes()
            raise DailyQuota(f"agent at {question['id']}: retry-after {last['retry_after']}")
        usage = turn.usage.as_dict()
        return {
            "reply": turn.reply, "error": turn.error, "tokens": usage, "requests": len(turn.llm_calls),
            "seconds": round(seconds, 2),
            "waited_s": round(sum(call["waited_ms"] for call in turn.llm_calls) / 1000, 2),
            "details": {"ended_by": turn.ended_by, "tools": [f"{s.tool}:{s.status}" for s in turn.steps],
                        "tool_arguments": [s.arguments for s in turn.steps], "repaired": turn.repaired,
                        "first_reply": turn.first_reply, "ungrounded": turn.ungrounded,
                        "year_corrected": turn.year_corrected, "expense_added": self._undo_writes()},
        }


class Gp1System:
    """One process per question: the order agent → GP1 is kept, and a crash cannot take the next questions with it."""

    def __init__(self, questions_file: Path, out: Path, env: dict[str, str], reserve: Callable[[], Any]) -> None:
        self.questions_file, self.out, self.env, self.reserve = questions_file, out, env, reserve

    def __call__(self, question: dict[str, Any]) -> dict[str, Any]:
        self.reserve()
        done = subprocess.run(
            [str(GP1_PYTHON), str(EVAL_DIR / "gp1_answer.py"), "--questions", str(self.questions_file),
             "--out", str(self.out), "--only", question["id"]],
            cwd=ROOT / "baseline", env=self.env, capture_output=True, text=True, encoding="utf-8")
        if done.returncode == DAILY_QUOTA_EXIT:
            raise DailyQuota(f"GP1 at {question['id']}: {done.stderr[-300:]}")
        if done.returncode != 0:
            raise RuntimeError(f"gp1_answer.py failed with code {done.returncode}: {done.stderr[-800:]}")
        found = next(item for item in read_items(self.out) if item["id"] == question["id"])
        calls = found["calls"]
        tokens = {"prompt": sum(c.get("prompt_tokens", 0) for c in calls),
                  "completion": sum(c.get("completion_tokens", 0) for c in calls),
                  "reasoning": sum(c.get("reasoning_tokens", 0) for c in calls),
                  "cached": sum(c.get("cached_tokens", 0) for c in calls)}
        tokens["total"] = tokens["prompt"] + tokens["completion"]
        return {"reply": found["reply"], "error": found["error"], "tokens": tokens, "requests": len(calls),
                "seconds": found["seconds"], "waited_s": found["waited_s"],
                "details": {"gp1_type": found["gp1_type"], "calls": calls, "expense_added": found["expense_added"]}}


# ---------- running ----------

def copy_databases(run_id: str, today: date) -> Path:
    """A copy per run and per day. A resumed run on the same day reuses it, with what was already undone in it."""
    target = GENERATED / "runs" / run_id / today.isoformat()
    if not target.exists():
        target.mkdir(parents=True)
        for name in ("gp1.db", "gp2.db"):
            shutil.copy2(GENERATED / name, target / name)
        for name in ("gp1_chroma", "gp2_chroma"):
            shutil.copytree(GENERATED / name, target / name)
    return target


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def new_config(run_id: str, settings: Settings, questions_file: Path, today: date) -> dict[str, Any]:
    return {
        "run_id": run_id, "model": settings.model_main, "reasoning_effort": EFFORT, "git_commit": git_commit(),
        "date": today.isoformat(), "questions_file": questions_file.relative_to(ROOT).as_posix(),
        "questions_sha256": hashlib.sha256(questions_file.read_bytes()).hexdigest(),
        "agent": {"memory_messages": 0, "corrections_enabled": False, "max_tokens": settings.llm_max_tokens,
                  "max_steps": settings.llm_max_steps, "grounding_repair": settings.grounding_repair},
        "gp1": {"max_tokens": GP1_MAX_TOKENS}, "segments": [],
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DATA_DIR / "pilot_numeric_questions.jsonl")
    parser.add_argument("--resume", help="the run_id of a stopped run")
    args = parser.parse_args(argv)

    today = date.today()
    try:
        manifest, records = load_database(GENERATED, today)
    except WrongDay as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1
    base = get_settings()
    if not base.groq_key_present:
        print("Stopped: GROQ_API_KEY is not set.", file=sys.stderr)
        return 1
    settings = base.model_copy(update={"agent_today": today, "reasoning_effort": EFFORT, "memory_messages": 0,
                                       "corrections_enabled": False})
    run_id = args.resume or f"rq2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder = RESULTS / run_id
    questions_file = args.questions.resolve()
    questions = load_questions(questions_file)
    if args.resume:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        config.pop("stopped", None)
    else:
        folder.mkdir(parents=True)
        config = new_config(run_id, settings, questions_file, today)
    config["segments"].append({"today": today.isoformat(), "records_sha256": manifest["records_sha256"],
                               "seed": manifest["seed"], "expenses": manifest["expenses"], "git_commit": git_commit()})

    def save_config() -> None:
        (folder / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    save_config()
    model = settings.model_main
    copies = copy_databases(run_id, today)
    ollama = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    warmup = ollama.warmup()
    if not warmup["ok"]:
        print(f"Stopped before any model call: Ollama is not ready ({warmup['error']}).", file=sys.stderr)
        return 1

    client = GroqClient.from_settings(settings)
    engine = make_engine(f"sqlite:///{(copies / 'gp2.db').as_posix()}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("BASELINE_")}
    env.update(PYTHONIOENCODING="utf-8", GROQ_API_KEY=settings.groq_api_key.get_secret_value(),
               BASELINE_DB_PATH=str(copies / "gp1.db"), BASELINE_CHROMA_DIR=str(copies / "gp1_chroma"),
               BASELINE_LLM_MODEL=model, BASELINE_REASONING_EFFORT=EFFORT)
    code = 0
    with make_session_factory(engine)() as session:
        store = ChromaVectorStore("", 0, client=chromadb.PersistentClient(path=str(copies / "gp2_chroma")))
        systems = {
            "agent": AgentSystem(session, settings, client, model, ollama, store, today),
            "gp1": Gp1System(questions_file, copies / "gp1_items.jsonl", env,
                             lambda: client.limiter(model).acquire(GP1_TOKENS_RESERVED)),
        }
        try:
            run_items(questions, records, manifest, systems, folder / "items.jsonl")
        except DailyQuota as exc:
            config["stopped"] = f"daily quota: {exc}"
            config["rate_limits"] = client.last_rate_limits.get(model)
            print(f"STOPPED: {exc}. Tomorrow: generate the database again, then --resume {run_id}", flush=True)
            code = DAILY_QUOTA_EXIT
    if "rate_limits" not in config:
        config["rate_limits"] = client.last_rate_limits.get(model)
    save_config()
    engine.dispose()
    client.close()
    print(json.dumps(write_summary(folder, config), ensure_ascii=False, indent=1), flush=True)
    print(f"saved {folder}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
