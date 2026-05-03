"""
LLM Parser — استخراج بيانات المصروفات من النص الطبيعي
Groq API + Llama 3.3 70B — مع retry و timeout محسّن
"""
import json, requests, time
from datetime import date, timedelta
from src.llm.prompts import EXPENSE_EXTRACTION_PROMPT


class ExpenseParser:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _call_llm(self, system_prompt: str, user_message: str, max_retries: int = 3) -> str:
        """إرسال رسالة للنموذج مع retry"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url, headers=self.headers,
                    json=payload, timeout=60
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 3
                    time.sleep(wait)
                    continue
                raise TimeoutError("❌ انتهت مهلة الاتصال بعد عدة محاولات.")

            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                raise ConnectionError("❌ لا يمكن الاتصال بـ Groq. تأكد من الإنترنت.")

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    wait = (attempt + 1) * 5
                    time.sleep(wait)
                    continue
                elif response.status_code == 401:
                    raise ValueError("❌ مفتاح API غير صالح.")
                raise ValueError(f"❌ خطأ: {e}")

    def _parse_json(self, text: str) -> dict:
        text = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return None

    def parse_expense(self, user_message: str) -> dict:
        today = date.today()
        yesterday = today - timedelta(days=1)
        prompt = EXPENSE_EXTRACTION_PROMPT.format(
            today=today.isoformat(), yesterday=yesterday.isoformat()
        )
        try:
            raw = self._call_llm(prompt, user_message)
        except Exception as e:
            return {"type": "error", "message": str(e)}

        result = self._parse_json(raw)
        if result is None:
            return {"type": "error", "message": "❌ لم أستطع فهم الرد.", "raw": raw}

        msg_type = result.get("type", "expense")
        if msg_type in ["query", "advice", "chat", "not_expense"]:
            return {"type": msg_type, "source_text": user_message}
        if msg_type == "expense":
            return self._validate_expense(result, user_message)
        return {"type": msg_type, "source_text": user_message}

    def _validate_expense(self, data: dict, source_text: str) -> dict:
        from src.llm.prompts import CATEGORIES_LIST
        result = {
            "type": "expense", "source_text": source_text,
            "amount": None, "category": "متفرقات",
            "date": date.today().isoformat(), "description": ""
        }
        amount = data.get("amount")
        if amount is not None:
            try:
                result["amount"] = float(amount)
            except (ValueError, TypeError):
                result["amount"] = None
        if result["amount"] is None or result["amount"] <= 0:
            return {"type": "need_amount", "source_text": source_text, "message": "🤔 كم المبلغ؟"}

        category = data.get("category", "متفرقات")
        if category in CATEGORIES_LIST:
            result["category"] = category
        else:
            for cat in CATEGORIES_LIST:
                if category in cat or cat in category:
                    result["category"] = cat
                    break

        expense_date = data.get("date")
        if expense_date:
            try:
                date.fromisoformat(expense_date)
                result["date"] = expense_date
            except (ValueError, TypeError):
                result["date"] = date.today().isoformat()

        result["description"] = data.get("description", "")
        return result

    def test_connection(self) -> bool:
        try:
            r = self._call_llm("أجب بكلمة: مرحبا", "جاهز؟", max_retries=2)
            print(f"✅ الاتصال بـ Groq ناجح! الرد: {r}")
            return True
        except Exception as e:
            print(f"❌ فشل الاتصال: {e}")
            return False
