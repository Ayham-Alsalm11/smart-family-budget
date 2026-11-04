"""
Chat Engine — المحرك الرئيسي للمحادثة
يربط: LLM Parser + RAG (SQL + Embedding) + Analysis
"""
import requests, json
from datetime import date, timedelta
from src.llm.parser import ExpenseParser
from src.llm.prompts import ADVICE_PROMPT, QUERY_RESPONSE_PROMPT, CHAT_PROMPT
from src.rag.retriever import FinancialRetriever
from src.analysis.spending import SpendingAnalyzer
from src.database.crud import add_expense


class ChatEngine:
    def __init__(self, api_key: str, family_id: int = 1):
        self.parser = ExpenseParser(api_key=api_key)
        self.retriever = FinancialRetriever(family_id=family_id)
        self.analyzer = SpendingAnalyzer(family_id=family_id)
        self.family_id = family_id
        self.api_key = api_key

    def process_message(self, user_message: str) -> dict:
        """معالجة رسالة المستخدم"""
        parsed = self.parser.parse_expense(user_message)
        msg_type = parsed.get("type", "error")

        if msg_type == "expense":
            return self._handle_expense(parsed)
        elif msg_type == "need_amount":
            return {"type": "need_amount", "response": "🤔 كم المبلغ؟", "data": parsed, "show_chart": False}
        elif msg_type == "query":
            return self._handle_query(user_message)
        elif msg_type == "advice":
            return self._handle_advice(user_message)
        elif msg_type == "chat":
            return self._handle_chat(user_message)
        elif msg_type == "error":
            return {"type": "error", "response": parsed.get("message", "حصل خطأ"), "data": {}, "show_chart": False}
        else:
            return self._handle_chat(user_message)

    def _handle_expense(self, parsed: dict) -> dict:
        """معالجة إدخال مصروف + تخزين embedding"""
        try:
            expense = add_expense(
                family_id=self.family_id,
                category_name=parsed["category"],
                amount=parsed["amount"],
                description=parsed.get("description", ""),
                source_text=parsed.get("source_text", ""),
                expense_date=date.fromisoformat(parsed["date"])
            )

            # === تخزين Embedding في ChromaDB ===
            self.retriever.store_expense_embedding(
                expense_id=expense.id,
                source_text=parsed.get("source_text", ""),
                amount=parsed["amount"],
                category=parsed["category"],
                expense_date=parsed["date"],
                description=parsed.get("description", "")
            )

            alerts = self.retriever.get_alerts()
            response = f"✅ تم تسجيل {parsed['amount']:,.0f} ل.س\n📂 {parsed['category']} | 📅 {parsed['date']}"
            if alerts:
                response += f"\n\n{alerts}"

            return {"type": "expense", "response": response, "data": parsed, "show_chart": False}
        except Exception as e:
            return {"type": "error", "response": f"❌ خطأ: {str(e)}", "data": {}, "show_chart": False}

    def _handle_query(self, user_message: str) -> dict:
        """معالجة استعلام — SQL + Embedding RAG"""
        context = self.retriever.get_query_context(query=user_message)
        prompt = QUERY_RESPONSE_PROMPT.format(user_data=context)
        try:
            response = self._call_llm(prompt, user_message)
        except Exception:
            response = context
        return {"type": "query", "response": response, "data": self.analyzer.get_chart_data(), "show_chart": True}

    def _handle_advice(self, user_message: str) -> dict:
        """معالجة طلب توصية — سياق كامل"""
        context = self.retriever.get_full_context(query=user_message)
        prompt = ADVICE_PROMPT.format(user_data=context)
        try:
            response = self._call_llm(prompt, user_message)
        except Exception:
            response = "❌ لم أستطع توليد التوصيات."
        return {"type": "advice", "response": response, "data": self.analyzer.get_chart_data(), "show_chart": True}

    def _handle_chat(self, user_message: str) -> dict:
        try:
            response = self._call_llm(CHAT_PROMPT, user_message)
        except Exception:
            response = "أهلاً! أنا ميزان 😊 اكتب مصاريفك وأنا بسجّلهم."
        return {"type": "chat", "response": response, "data": {}, "show_chart": False}

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7, "max_tokens": 500
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload, timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
