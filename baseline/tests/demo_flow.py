"""
عرض تجريبي — محادثة تفاعلية
$env:GROQ_API_KEY="gsk_xxx"
python -m tests.demo_flow
"""
import os, sys
from datetime import date
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chat.engine import ChatEngine
from src.analysis.spending import SpendingAnalyzer
from config import GROQ_API_KEY


def demo():
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if api_key == "YOUR_API_KEY_HERE":
        print("⚠️ أدخل مفتاح Groq API!")
        return

    engine = ChatEngine(api_key=api_key)

    print("=" * 55)
    print("💰 ميزان — مساعد الميزانية العائلي الذكي")
    print("   اكتب مصاريفك أو اسأل عنها")
    print("   اكتب 'خروج' للإنهاء")
    print("=" * 55 + "\n")

    while True:
        user_input = input("👤 أنت: ").strip()
        if not user_input:
            continue
        if user_input in ["خروج", "exit", "quit"]:
            print("👋 مع السلامة!")
            break

        result = engine.process_message(user_input)
        print(f"💰 ميزان: {result['response']}\n")


if __name__ == "__main__":
    demo()
