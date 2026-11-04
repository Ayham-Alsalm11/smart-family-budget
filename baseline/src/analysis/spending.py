"""
Analysis — تحليل الإنفاق
"""
from datetime import date
from src.database.crud import get_family, get_total_spending, get_expenses_by_category


class SpendingAnalyzer:
    def __init__(self, family_id: int = 1):
        self.family_id = family_id

    def get_monthly_report(self, year=None, month=None) -> dict:
        today = date.today()
        year = year or today.year
        month = month or today.month
        family = get_family(self.family_id)
        if not family:
            return {"error": "لا توجد بيانات"}

        income = family.monthly_income
        total = get_total_spending(self.family_id, year, month)
        by_cat = get_expenses_by_category(self.family_id, year, month)

        categories = []
        over_budget = []
        for name_ar, name_en, budget_pct, cat_total, count in (by_cat or []):
            limit = income * budget_pct / 100
            usage = (cat_total / limit * 100) if limit > 0 else 0
            cat = {"name_ar": name_ar, "name_en": name_en, "total": cat_total,
                   "count": count, "budget_limit": limit, "usage_percent": usage}
            categories.append(cat)
            if usage > 100:
                over_budget.append(cat)

        return {
            "month": month, "year": year, "income": income,
            "total_spending": total, "remaining": income - total,
            "spending_percent": (total / income * 100) if income > 0 else 0,
            "categories": categories, "over_budget": over_budget
        }

    def get_chart_data(self, year=None, month=None) -> dict:
        today = date.today()
        year = year or today.year
        month = month or today.month
        by_cat = get_expenses_by_category(self.family_id, year, month)

        labels, values = [], []
        colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF",
                  "#FF9F40", "#C9CBCF", "#7BC8A4", "#E7E9ED", "#F7464A",
                  "#46BFBD", "#FDB45C"]
        for name_ar, _, _, cat_total, _ in (by_cat or []):
            labels.append(name_ar)
            values.append(cat_total)
        return {"labels": labels, "values": values, "colors": colors[:len(labels)]}

    def get_comparison(self) -> dict:
        today = date.today()
        prev_m, prev_y = (12, today.year - 1) if today.month == 1 else (today.month - 1, today.year)
        cur = get_total_spending(self.family_id, today.year, today.month)
        prev = get_total_spending(self.family_id, prev_y, prev_m)
        change = ((cur - prev) / prev * 100) if prev > 0 else 0
        return {
            "current_month": today.month, "current_total": cur,
            "prev_month": prev_m, "prev_total": prev,
            "change_percent": change,
            "trend": "زيادة" if change > 0 else "انخفاض" if change < 0 else "ثبات"
        }
