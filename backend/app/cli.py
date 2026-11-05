"""Terminal chat for development only (§1).

    cd backend
    .venv\\Scripts\\python -m app.cli --user manager --model small [--today 2026-11-04] [--db data\\dev.db]

The member is chosen by username from the seeded database, WITHOUT a password. This is a
development shortcut; real users log in through the API (§8).
"""
import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from typing import Any, TextIO

from sqlalchemy import select

from app.agent.context import ToolContext
from app.agent.loop import TurnOptions, TurnResult, run_turn
from app.db.models import Member
from app.db.session import init_db, make_engine, make_session_factory
from app.llm.client import GroqClient
from app.rag.embeddings import OllamaEmbedder
from app.rag.vector_store import ChromaVectorStore
from app.settings import get_settings

EXIT_WORDS = {"exit", "quit", "خروج"}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Development chat with the agent.")
    parser.add_argument("--user", required=True, help="username from the seed, e.g. manager or member")
    parser.add_argument("--model", choices=["main", "small"], default="small")
    parser.add_argument("--today", type=date.fromisoformat, help="fix today's date (YYYY-MM-DD)")
    parser.add_argument("--db", help="SQLite file (default: DATABASE_URL)")
    return parser.parse_args(argv)


def _rag(settings: Any, out: TextIO) -> tuple[Any, Any]:
    """Ollama and ChromaDB when both are reachable; otherwise search_expenses reports not_available."""
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    try:
        store = ChromaVectorStore(settings.chroma_host, settings.chroma_port)
        store.heartbeat()
    except Exception:
        print("(ChromaDB is not reachable: search by meaning is off)", file=out)
        return None, None
    if not embedder.status()["ok"]:
        print("(Ollama is not reachable: search by meaning is off)", file=out)
        return None, None
    return embedder, store


def print_turn(turn: TurnResult, out: TextIO) -> None:
    print(f"ميزان: {turn.reply}", file=out)
    for step in turn.steps:
        arguments = json.dumps(step.arguments, ensure_ascii=False)
        print(f"  [{step.step}] {step.tool} {arguments} → {step.status}", file=out)
    if turn.ungrounded:
        print(f"  ungrounded numbers: {', '.join(turn.ungrounded)}", file=out)
    u = turn.usage
    print(f"  tokens: prompt {u.prompt_tokens}, completion {u.completion_tokens} (reasoning {u.reasoning_tokens}), "
          f"total {u.total_tokens} in {turn.llm_steps} LLM step(s) [{turn.ended_by}]", file=out)


def main(argv: list[str] | None = None, client: Any = None, input_fn: Callable[[str], str] = input,
         out: TextIO | None = None, rag: tuple[Any, Any] | None = None) -> int:
    """`client` and `rag` (embedder, vector store) can be injected by tests."""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    out = out or sys.stdout
    args = _parse(argv)
    settings = get_settings()
    engine = make_engine(f"sqlite:///{args.db}" if args.db else settings.database_url)
    init_db(engine)
    session = make_session_factory(engine)()
    member = session.scalar(select(Member).where(Member.username == args.user, Member.is_active.is_(True)))
    if member is None:
        print(f"No active user {args.user!r}. Run: python -m app.db.seed", file=sys.stderr)
        return 1

    client = client or GroqClient.from_settings(settings)
    embedder, store = rag if rag is not None else _rag(settings, out)
    model = settings.model_id(args.model)
    ctx = ToolContext(member.family_id, member.id, member.role, args.today or settings.today(), model)
    print(f"{member.name} ({member.role}) · {model} · reasoning {settings.reasoning_effort} · today {ctx.today}. "
          f"Type 'exit' to stop.", file=out)
    while True:
        try:
            message = input_fn("أنت: ").strip()
        except EOFError:
            break
        if not message:
            continue
        if message.lower() in EXIT_WORDS:
            break
        print_turn(run_turn(session, ctx, message, client, embedder=embedder, vector_store=store,
                            max_steps=settings.llm_max_steps, options=TurnOptions.from_settings(settings)), out)
    session.close()
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
