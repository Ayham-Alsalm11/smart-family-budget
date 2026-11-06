"""Evaluation database generator (§9.2): one neutral records file, and from it the GP2 and the GP1 databases.

    backend\\.venv\\Scripts\\python -m eval.generate_eval_db [--today 2026-11-06] [--seed 42] [--overwrite]

The records file comes first, so that both systems get the same expenses and the reference answers (eval/questions.py)
can be computed without either system's code. Fairness (§9.2): one currency, the base currency, because GP1 has no
currencies; no budget rows, because GP1 ignores that table (P6), so a limit is income × budget_percent in both systems.

--today defaults to the real date, because GP1 uses date.today(): the experiment has to run on the day of the database.
Needs Ollama. ChromaDB runs in-process here (PersistentClient), so the development server and its collections are not
touched. GP1's side is built by eval/gp1_build.py, in GP1's own environment and with its own embedding code.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from eval import EVAL_DIR, GENERATED, GP1_PYTHON, ROOT

import chromadb  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.db import seed  # noqa: E402
from app.db.categories import CATEGORIES  # noqa: E402
from app.db.models import Budget, Category, Expense, Family, Member  # noqa: E402
from app.db.session import init_db, make_engine, make_session_factory  # noqa: E402
from app.rag.embeddings import OllamaEmbedder  # noqa: E402
from app.rag.vector_store import ChromaVectorStore  # noqa: E402
from app.services.expenses import reindex_missing  # noqa: E402
from app.settings import get_settings  # noqa: E402

SEED = 42
MONTHS = 12
FAMILY = "عائلة التقييم"
INCOME = 500_000  # old SYP, as in GP1's family
CURRENCY = "SYP_OLD"
RENT = 125_000
SATURDAY = 5
DRAWS_PER_MONTH = 18  # with the fixed expenses, a month comes close to the income
OUTPUTS = ("records.json", "manifest.json", "gp2.db", "gp2_chroma", "gp1.db", "gp1_chroma")
MEMBERS = [{"username": "manager", "name": "أبو سامر", "role": "manager"},
           {"username": "member", "name": "أم سامر", "role": "member"}]
# (day of the month, description, lowest, highest): the same bills come every month.
MONTHLY_BILLS = [(3, "اشتراك النت الشهري", 8_000, 12_000), (5, "فاتورة الكهربا", 10_000, 30_000),
                 (10, "فاتورة المي", 5_000, 12_000)]
GROCERIES = "أغراض البيت من السوق"
BILLS = "فواتير"
FOOD = "أكل وشرب"
RENT_CATEGORY = "إيجار"


def _month_back(today: date, months: int) -> date:
    index = today.year * 12 + today.month - 1 - months
    return date(index // 12, index % 12 + 1, 1)


def _month_days(first: date) -> list[date]:
    days = []
    day = first
    while day.month == first.month:
        days.append(day)
        day += timedelta(days=1)
    return days


def _source_text(amount: int, description: str) -> str:
    """Both systems embed the user's original words, so the generated expenses need some."""
    return f"دفعت {amount} {description}"


def generate_records(today: date, seed_value: int = SEED) -> dict[str, Any]:
    """The neutral records: the current month up to `today` and the 11 months before it, oldest first."""
    rng = random.Random(seed_value)
    names = list(seed.DEMO_SPENDING)
    weights = [seed.DEMO_SPENDING[name][2] for name in names]
    usernames = [member["username"] for member in MEMBERS]
    drawn = []

    def add(when: date, category: str, amount: int, description: str, who: str) -> None:
        drawn.append({"member": who, "category": category, "amount": amount, "currency": CURRENCY,
                      "date": when, "description": description, "source_text": _source_text(amount, description)})

    for back in range(MONTHS - 1, -1, -1):
        days = _month_days(_month_back(today, back))
        add(days[0], RENT_CATEGORY, RENT, seed.RENT_DESCRIPTION, "manager")
        for day, description, low, high in MONTHLY_BILLS:
            add(days[day - 1], BILLS, rng.randrange(low, high + 1, 500), description, "manager")
        for day in days:
            if day.weekday() == SATURDAY:
                add(day, FOOD, rng.randrange(15_000, 30_001, 500), GROCERIES, rng.choice(usernames))
        for _ in range(DRAWS_PER_MONTH):
            category = rng.choices(names, weights)[0]
            low, high, _weight = seed.DEMO_SPENDING[category]
            add(rng.choice(days), category, rng.randrange(low, high + 1, 500),
                rng.choice(seed.DEMO_DESCRIPTIONS[category]), rng.choice(usernames))

    # The rest of the current month is drawn and dropped, so that every draw before it stays the same.
    kept = sorted((item for item in drawn if item["date"] <= today), key=lambda item: item["date"])
    expenses = [{"id": number, **item, "date": item["date"].isoformat()} for number, item in enumerate(kept, start=1)]
    return {
        "today": today.isoformat(), "seed": seed_value,
        "family": {"name": FAMILY, "monthly_income": INCOME, "currency": CURRENCY},
        "members": MEMBERS,
        "categories": [{"name": c["name_ar"], "budget_percent": c["budget_percent"]} for c in CATEGORIES],
        "expenses": expenses,
    }


def write_records(records: dict[str, Any], path: Path) -> str:
    """Returns the SHA-256 of the file. Written as bytes: the hash must not depend on the platform's line endings."""
    data = json.dumps(records, ensure_ascii=False, indent=1).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_gp2(records: dict[str, Any], folder: Path, embedder: Any) -> dict[str, int]:
    """The GP2 schema plus its evaluation collections. Returns how many expenses and vectors were stored."""
    engine = make_engine(f"sqlite:///{(folder / 'gp2.db').as_posix()}")
    init_db(engine)
    with make_session_factory(engine)() as session:
        seed.seed_categories(session)
        family = Family(name=records["family"]["name"], monthly_income=Decimal(records["family"]["monthly_income"]),
                        base_currency=records["family"]["currency"], usd_rate=None)
        session.add(family)
        session.flush()
        members = {}
        for item in records["members"]:
            # Nobody logs in to this database, so the hash is one that no password can match.
            members[item["username"]] = Member(family_id=family.id, name=item["name"], username=item["username"],
                                               password_hash="!", role=item["role"])
            session.add(members[item["username"]])
        session.flush()
        categories = {c.name_ar: c for c in session.scalars(select(Category))}
        for item in records["expenses"]:
            amount = Decimal(item["amount"])
            session.add(Expense(
                id=item["id"], family_id=family.id, member_id=members[item["member"]].id,
                category_id=categories[item["category"]].id, amount=amount, currency=item["currency"],
                amount_base=amount, rate_to_base=Decimal(1), description=item["description"],
                source_text=item["source_text"], expense_date=date.fromisoformat(item["date"]),
            ))
        session.commit()

        store = ChromaVectorStore("", 0, client=chromadb.PersistentClient(path=str(folder / "gp2_chroma")))
        store.ensure_collections()
        vectors = reindex_missing(session, embedder, store)
        expenses = session.scalar(select(func.count()).select_from(Expense))
        if session.scalar(select(func.count()).select_from(Budget)):
            raise RuntimeError("the evaluation database must have no budget rows (§9.2)")
    engine.dispose()
    if vectors != expenses:
        raise RuntimeError(f"only {vectors} of {expenses} expenses were embedded")
    return {"expenses": expenses, "vectors": vectors}


def build_gp1(folder: Path) -> dict[str, int]:
    """Runs eval/gp1_build.py in GP1's environment; its last output line is a JSON summary."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "BASELINE_DB_PATH": str(folder / "gp1.db"),
           "BASELINE_CHROMA_DIR": str(folder / "gp1_chroma")}
    env.pop("GROQ_API_KEY", None)  # building needs no model
    done = subprocess.run([str(GP1_PYTHON), str(EVAL_DIR / "gp1_build.py"), str(folder / "records.json")],
                          cwd=ROOT / "baseline", env=env, capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        raise RuntimeError(f"gp1_build.py failed:\n{done.stdout[-1500:]}\n{done.stderr[-1500:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def generate(today: date, seed_value: int, folder: Path, overwrite: bool, embedder: Any) -> dict[str, Any]:
    existing = [name for name in OUTPUTS if (folder / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{folder} already has {', '.join(existing)}; use --overwrite to replace them")
    folder.mkdir(parents=True, exist_ok=True)
    for name in existing:
        target = folder / name
        shutil.rmtree(target) if target.is_dir() else target.unlink()

    records = generate_records(today, seed_value)
    digest = write_records(records, folder / "records.json")
    gp2 = build_gp2(records, folder, embedder)
    gp1 = build_gp1(folder)
    if gp1 != gp2:
        raise RuntimeError(f"the two databases differ: GP1 {gp1}, GP2 {gp2}")
    manifest = {"today": today.isoformat(), "seed": seed_value, "expenses": len(records["expenses"]),
                "records_sha256": digest, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "gp1": gp1, "gp2": gp2}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=GENERATED)
    parser.add_argument("--overwrite", action="store_true", help="replace an earlier generated database")
    args = parser.parse_args(argv)

    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    warmup = embedder.warmup()
    if not warmup["ok"]:
        print(f"Stopped before writing anything: Ollama is not ready ({warmup['error']}).", file=sys.stderr)
        return 1
    try:
        manifest = generate(args.today, args.seed, args.out.resolve(), args.overwrite, embedder)
    except FileExistsError as exc:
        print(f"Stopped: {exc}.", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
