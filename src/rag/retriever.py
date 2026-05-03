"""
RAG Retriever — استرجاع بيانات المستخدم المالية
يستخدم: SQL (بيانات منظمة) + BGE-M3 Embedding (بحث دلالي)
"""
from datetime import date
from src.database.crud import (
    get_family, get_total_spending, get_expenses_by_category,
    get_recent_expenses
)
from src.rag.embedding import EmbeddingEngine


class FinancialRetriever:
    """استرجاع البيانات المالية — SQL + RAG"""

    def __init__(self, family_id: int = 1):
        self.family_id = family_id
        # محاولة تشغيل Embedding Engine
        try:
            self.embedding = EmbeddingEngine()
            self.embedding_available = self.embedding.test_connection()
        except Exception:
            self.embedding = None
            self.embedding_available = False
            print("⚠️ BGE-M3 غير متاح — سيتم استخدام SQL فقط")

    def store_expense_embedding(self, expense_id: int, source_text: str,
                                 amount: float, category: str,
                                 expense_date: str, description: str):
        """تخزين embedding لمصروف جديد"""
        if not self.embedding_available:
            return

        # النص اللي بنحوّله لـ vector
        text = f"{description} {category} {amount} {source_text}"

        self.embedding.add_expense(
            expense_id=expense_id,
            text=text,
            metadata={
                "amount": amount,
                "category": category,
                "date": expense_date,
                "description": description,
                "family_id": self.family_id
            }
        )

    def get_current_month_summary(self) -> str:
        """ملخص مصاريف الشهر الحالي (من SQL)"""
        today = date.today()
        family = get_family(self.family_id)
        if not family:
            return "لا توجد بيانات عائلة."

        income = family.monthly_income
        total = get_total_spending(self.family_id, today.year, today.month)
        remaining = income - total
        spent_pct = (total / income * 100) if income > 0 else 0
        days_left = self._days_remaining()

        summary = f"""=== ملخص الشهر ({today.month}/{today.year}) ===
الدخل: {income:,.0f} ل.س
المصروفات: {total:,.0f} ل.س ({spent_pct:.0f}%)
المتبقي: {remaining:,.0f} ل.س
أيام متبقية: {days_left}
"""
        by_cat = get_expenses_by_category(self.family_id, today.year, today.month)
        if by_cat:
            summary += "\nتفصيل الفئات:\n"
            for name_ar, _, budget_pct, cat_total, count in by_cat:
                limit = income * budget_pct / 100
                usage = (cat_total / limit * 100) if limit > 0 else 0
                status = "🚨" if usage >= 100 else "⚠️" if usage >= 80 else "✅"
                summary += f"  {status} {name_ar}: {cat_total:,.0f} ({usage:.0f}% من الحد) — {count} عملية\n"
        return summary

    def get_alerts(self) -> str:
        """تنبيهات تجاوز الميزانية"""
        today = date.today()
        family = get_family(self.family_id)
        if not family:
            return ""

        income = family.monthly_income
        by_cat = get_expenses_by_category(self.family_id, today.year, today.month)
        alerts = []
        for name_ar, _, budget_pct, cat_total, _ in (by_cat or []):
            limit = income * budget_pct / 100
            if limit > 0:
                usage = cat_total / limit * 100
                if usage >= 100:
                    alerts.append(f"🚨 {name_ar}: تجاوزت الحد! ({cat_total:,.0f}/{limit:,.0f})")
                elif usage >= 80:
                    alerts.append(f"⚠️ {name_ar}: {usage:.0f}% من الحد ({cat_total:,.0f}/{limit:,.0f})")

        return "\n=== تنبيهات ===\n" + "\n".join(alerts) + "\n" if alerts else ""

    def get_recent_transactions(self, limit: int = 5) -> str:
        """آخر المعاملات"""
        expenses = get_recent_expenses(self.family_id, limit)
        if not expenses:
            return "لا توجد مصاريف مسجلة."
        text = f"\n=== آخر {limit} مصاريف ===\n"
        for exp in expenses:
            text += f"  #{exp.id} | {exp.expense_date} | {exp.amount:,.0f} ل.س | {exp.description}\n"
        return text

    def get_semantic_context(self, query: str) -> str:
        """استرجاع سياق بالـ Embedding (RAG)"""
        if not self.embedding_available:
            return ""
        return self.embedding.get_context_for_query(query)

    def get_full_context(self, query: str = "") -> str:
        """السياق الكامل: SQL + Embedding"""
        context = self.get_current_month_summary()
        context += self.get_alerts()
        context += self.get_recent_transactions()
        # إضافة نتائج البحث الدلالي إذا متاح
        if query and self.embedding_available:
            semantic = self.get_semantic_context(query)
            if semantic:
                context += f"\n{semantic}"
        return context

    def get_query_context(self, query: str = "") -> str:
        """سياق مختصر للاستعلامات"""
        context = self.get_current_month_summary()
        context += self.get_alerts()
        if query and self.embedding_available:
            semantic = self.get_semantic_context(query)
            if semantic:
                context += f"\n{semantic}"
        return context

    def _days_remaining(self) -> int:
        today = date.today()
        if today.month == 12:
            next_m = date(today.year + 1, 1, 1)
        else:
            next_m = date(today.year, today.month + 1, 1)
        return (next_m - today).days
