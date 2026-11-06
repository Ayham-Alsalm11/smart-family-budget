"""Builds GP1's evaluation database and ChromaDB from the neutral records (§9.2), with GP1's own models and embedding.

Started by eval/generate_eval_db.py, in the root venv with baseline/ as the working directory, BASELINE_DB_PATH and
BASELINE_CHROMA_DIR set. `store_expense_embedding` is GP1's own, so the embedded text and the metadata have its format
(amount as a float, every metadata value as text). Without both variables it stops, so baseline/data/ cannot be written.
"""
import json
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.getcwd())

if not (os.environ.get("BASELINE_DB_PATH") and os.environ.get("BASELINE_CHROMA_DIR")):
    sys.exit("BASELINE_DB_PATH and BASELINE_CHROMA_DIR are required: GP1's real data must stay as it is")

from src.database.models import Category, Expense, Family, get_session, init_db  # noqa: E402
from src.database.seed import seed_categories  # noqa: E402
from src.rag.retriever import FinancialRetriever  # noqa: E402


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as file:
        records = json.load(file)

    init_db()
    seed_categories()
    session = get_session()
    family = Family(name=records["family"]["name"], monthly_income=float(records["family"]["monthly_income"]),
                    currency="SYP")
    session.add(family)
    session.commit()
    family_id = family.id
    if family_id != 1:
        sys.exit(f"GP1 always reads family 1, and this family has id {family_id}")

    categories = {c.name_ar: c.id for c in session.query(Category).all()}
    for item in records["expenses"]:
        day = date.fromisoformat(item["date"])
        # created_at orders GP1's "recent expenses", so it follows the expense and not the moment of this script.
        session.add(Expense(id=item["id"], family_id=family_id, category_id=categories[item["category"]],
                            amount=float(item["amount"]), description=item["description"],
                            source_text=item["source_text"], expense_date=day,
                            created_at=datetime(day.year, day.month, day.day, 12)))
    session.commit()
    expenses = session.query(Expense).count()
    session.close()

    retriever = FinancialRetriever(family_id=family_id)
    if not retriever.embedding_available or retriever.embedding.get_embedding("تجربة") is None:
        sys.exit("Ollama is not available to GP1: an expense without its vector would make the databases differ")
    for item in records["expenses"]:
        retriever.store_expense_embedding(
            expense_id=item["id"], source_text=item["source_text"], amount=float(item["amount"]),
            category=item["category"], expense_date=item["date"], description=item["description"])
    vectors = retriever.embedding.collection.count()
    print(json.dumps({"expenses": expenses, "vectors": vectors}))
    return 0 if vectors == expenses else 1


if __name__ == "__main__":
    raise SystemExit(main())
