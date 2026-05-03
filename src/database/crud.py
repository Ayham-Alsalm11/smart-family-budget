"""
CRUD Operations — عمليات إنشاء/قراءة/تعديل/حذف
"""
from datetime import date, datetime
from sqlalchemy import func, extract
from sqlalchemy.orm import sessionmaker
from src.database.models import get_engine, Family, Category, Expense, Budget

_Session = sessionmaker(bind=get_engine(), expire_on_commit=False)


def _get_session():
    return _Session()


# ==================== Family ====================

def create_family(name: str, monthly_income: float, currency: str = "SYP") -> Family:
    session = _get_session()
    family = Family(name=name, monthly_income=monthly_income, currency=currency)
    session.add(family)
    session.commit()
    session.close()
    print(f"✅ تم إنشاء عائلة '{name}' بدخل {monthly_income:,.0f} {currency}")
    return family


def get_family(family_id: int = 1) -> Family:
    session = _get_session()
    family = session.query(Family).filter_by(id=family_id).first()
    session.close()
    return family


def update_income(family_id: int, new_income: float):
    session = _get_session()
    family = session.query(Family).filter_by(id=family_id).first()
    if family:
        family.monthly_income = new_income
        session.commit()
        print(f"✅ تم تحديث الدخل إلى {new_income:,.0f}")
    session.close()


# ==================== Category ====================

def get_all_categories():
    session = _get_session()
    categories = session.query(Category).all()
    session.close()
    return categories


def get_category_by_name(name_ar: str):
    session = _get_session()
    category = session.query(Category).filter_by(name_ar=name_ar).first()
    session.close()
    return category


# ==================== Expense ====================

def add_expense(family_id: int, category_name: str, amount: float,
                description: str = "", source_text: str = "",
                expense_date: date = None) -> Expense:
    session = _get_session()
    category = session.query(Category).filter_by(name_ar=category_name).first()
    if not category:
        category = session.query(Category).filter_by(name_ar="متفرقات").first()

    expense = Expense(
        family_id=family_id,
        category_id=category.id,
        amount=amount,
        description=description,
        source_text=source_text,
        expense_date=expense_date or date.today()
    )
    session.add(expense)
    session.commit()
    exp_id = expense.id
    cat_name = category.name_ar
    session.close()
    print(f"✅ تم تسجيل {amount:,.0f} — {cat_name}")
    return expense


def get_expenses_by_month(family_id: int, year: int, month: int):
    session = _get_session()
    expenses = session.query(Expense).filter(
        Expense.family_id == family_id,
        extract('year', Expense.expense_date) == year,
        extract('month', Expense.expense_date) == month
    ).order_by(Expense.expense_date.desc()).all()
    session.close()
    return expenses


def get_expenses_by_category(family_id: int, year: int, month: int):
    session = _get_session()
    results = session.query(
        Category.name_ar, Category.name_en, Category.budget_percent,
        func.sum(Expense.amount).label("total"),
        func.count(Expense.id).label("count")
    ).join(Category).filter(
        Expense.family_id == family_id,
        extract('year', Expense.expense_date) == year,
        extract('month', Expense.expense_date) == month
    ).group_by(Category.id).all()
    session.close()
    return results


def get_total_spending(family_id: int, year: int, month: int) -> float:
    session = _get_session()
    total = session.query(func.sum(Expense.amount)).filter(
        Expense.family_id == family_id,
        extract('year', Expense.expense_date) == year,
        extract('month', Expense.expense_date) == month
    ).scalar()
    session.close()
    return total or 0.0


def get_recent_expenses(family_id: int, limit: int = 10):
    session = _get_session()
    expenses = session.query(Expense).filter(
        Expense.family_id == family_id
    ).order_by(Expense.created_at.desc()).limit(limit).all()
    session.close()
    return expenses


def delete_expense(expense_id: int):
    session = _get_session()
    expense = session.query(Expense).filter_by(id=expense_id).first()
    if expense:
        session.delete(expense)
        session.commit()
        print(f"✅ تم حذف المصروف #{expense_id}")
    session.close()


def update_expense(expense_id: int, **kwargs):
    session = _get_session()
    expense = session.query(Expense).filter_by(id=expense_id).first()
    if expense:
        for key, value in kwargs.items():
            if hasattr(expense, key):
                setattr(expense, key, value)
        session.commit()
        print(f"✅ تم تعديل المصروف #{expense_id}")
    session.close()
