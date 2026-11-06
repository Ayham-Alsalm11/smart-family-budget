"""GP1's side of RQ2 (§9.4): answers the questions with ChatEngine.process_message, in GP1's own environment.

Started by eval/run_rq2.py, from inside baseline/ with the root venv, PYTHONIOENCODING=utf-8, GROQ_API_KEY and the four
BASELINE_* variables. By hand: python ..\\eval\\gp1_answer.py --questions <jsonl> --out <jsonl> [--only P01,P02]

One line per question is written as soon as it ends, and questions already in the file are skipped, so a stopped run can
continue. Any exception (such as A1) is recorded as a failure, as in GP1's own evaluation. A question that GP1 takes for
an expense adds a row: it is removed again, because the reference answers assume the same records for every question.
Exit code 2: the daily quota was reached, and the question it hit is left for the next run.
"""
import argparse
import json
import os
import sys
import time
from typing import Any

import gp1_http

DAILY_QUOTA_EXIT = 2


def answer(engine: Any, log: gp1_http.RequestLog, question: dict[str, Any], undo: Any = None) -> dict[str, Any]:
    log.calls.clear()
    started = time.perf_counter()
    reply, kind, error = "", "exception", None
    try:
        result = engine.process_message(question["question"])
        reply, kind = result["response"], result["type"]
    except Exception as exc:  # A1 and anything else: a failure of the system, not of the experiment
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started
    if error is None:
        error = gp1_http.failure(log.calls) or (reply if kind == "error" else None)
    return {
        "id": question["id"], "reply": reply, "gp1_type": kind, "error": error, "calls": list(log.calls),
        "seconds": round(seconds, 2), "waited_s": round(sum(call.get("waited_s", 0) for call in log.calls), 2),
        "expense_added": undo() if undo else [],
    }


def _undo_writes(engine: Any, newest_id: int, session_factory: Any, expense_model: Any) -> Any:
    def undo() -> list[float]:
        session = session_factory()
        added = session.query(expense_model).filter(expense_model.id > newest_id).all()
        amounts, ids = [row.amount for row in added], [str(row.id) for row in added]
        for row in added:
            session.delete(row)
        session.commit()
        session.close()
        if ids:
            engine.retriever.embedding.collection.delete(ids=ids)
        return amounts
    return undo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", help="comma-separated question ids")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    log = gp1_http.RequestLog()
    log.install()
    from sqlalchemy import func  # noqa: E402

    from src.chat.engine import ChatEngine  # noqa: E402
    from src.database.models import Expense, get_session  # noqa: E402

    engine = ChatEngine(api_key=os.environ["GROQ_API_KEY"], family_id=1)
    # GP1 skips the search silently when its first embedding times out (A3), and that is not what RQ2 measures.
    if not engine.retriever.embedding_available or engine.retriever.embedding.get_embedding("تجربة") is None:
        print("Stopped: Ollama is not ready for GP1.", file=sys.stderr)
        return 1

    with open(args.questions, encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [question for question in questions if question["id"] in wanted]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as file:
            done = {json.loads(line)["id"] for line in file if line.strip()}

    session = get_session()
    newest_id = session.query(func.max(Expense.id)).scalar()
    session.close()
    undo = _undo_writes(engine, newest_id, get_session, Expense)

    for question in questions:
        if question["id"] in done:
            continue
        item = answer(engine, log, question, undo)
        if log.daily_quota:
            print(f"Stopped at {question['id']}: the daily quota was reached.", file=sys.stderr)
            print(json.dumps({"stopped": "daily_quota", "limits": log.limits}), file=sys.stderr)
            return DAILY_QUOTA_EXIT
        with open(args.out, "a", encoding="utf-8") as file:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
