"""The development terminal chat (§1)."""
import io

from app import cli
from app.db import seed
from app.db.session import init_db, make_engine, make_session_factory

from conftest import TODAY
from fake_llm import FakeLLM, call, respond


def seeded(tmp_path) -> str:
    path = (tmp_path / "cli.db").as_posix()
    engine = make_engine(f"sqlite:///{path}")
    init_db(engine)
    with make_session_factory(engine)() as session:
        seed.seed_categories(session)
        seed.seed_demo_family(session, "a", "b", TODAY)
    engine.dispose()
    return path


def test_cli_prints_reply_steps_and_tokens(tmp_path):
    llm = FakeLLM([respond(call("get_total", period="this_month")), respond(content="صرفتو هالشهر شي معقول.")])
    inputs = iter(["كم صرفنا هالشهر؟", "exit"])
    out = io.StringIO()
    code = cli.main(["--user", "member", "--model", "small", "--today", TODAY.isoformat(), "--db", seeded(tmp_path)],
                    client=llm, input_fn=lambda _prompt: next(inputs), out=out, rag=(None, None))
    text = out.getvalue()
    assert code == 0
    assert "ميزان: صرفتو هالشهر شي معقول." in text
    assert '[1] get_total {"period": "this_month"} → ok' in text
    assert "tokens: prompt 200, completion 40 (reasoning 10), total 240 in 2 LLM step(s) [reply]" in text


def test_cli_unknown_user(tmp_path):
    code = cli.main(["--user", "nobody", "--db", seeded(tmp_path)], client=FakeLLM([]), out=io.StringIO(),
                    rag=(None, None))
    assert code == 1
