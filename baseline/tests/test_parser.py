"""
اختبار LLM Parser
GROQ_API_KEY=gsk_xxx python -m tests.test_parser
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm.parser import ExpenseParser
from config import GROQ_API_KEY


def test_parser():
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if api_key == "YOUR_API_KEY_HERE":
        print("=" * 55)
        print("⚠️  مطلوب مفتاح Groq API!")
        print("  PowerShell: $env:GROQ_API_KEY=\"gsk_xxx\"")
        print("  CMD:        set GROQ_API_KEY=gsk_xxx")
        print("  ثم: python -m tests.test_parser")
        print("=" * 55)
        return

    parser = ExpenseParser(api_key=api_key)
    print("=" * 55)
    print("🔌 اختبار الاتصال بـ Groq...")
    print("=" * 55)
    if not parser.test_connection():
        return

    tests = [
        ("دفعت 5000 بنزين", "expense"),
        ("اشتريت خضرا بـ 3000", "expense"),
        ("فاتورة الكهربا 15000", "expense"),
        ("رحت عالدكتور دفعت 20 ألف", "expense"),
        ("دفعت إيجار البيت 150 ألف", "expense"),
        ("اشتريت شي من الدكان بخمس تلاف", "expense"),
        ("أخدت تكسي بألفين", "expense"),
        ("دفعت أقساط الجامعة 50 ألف", "expense"),
        ("رحت عالسوق", "need_amount"),
        ("كم صرفت هالشهر؟", "query"),
        ("وين رايحة مصاريفي؟", "query"),
        ("مرحبا كيفك", "not_expense"),
    ]

    print("\n" + "=" * 55)
    print("🧪 اختبار استخراج المصروفات")
    print("=" * 55 + "\n")

    passed = 0
    for msg, expected in tests:
        result = parser.parse_expense(msg)
        actual = result.get("type", "unknown")
        ok = "✅" if actual == expected else "❌"
        if actual == expected:
            passed += 1
        print(f"  {ok} \"{msg}\"")
        print(f"     النوع: {actual} (متوقع: {expected})")
        if actual == "expense":
            print(f"     💰 {result.get('amount', '?'):,.0f} | 📂 {result.get('category', '?')} | 📅 {result.get('date', '?')}")
        print()

    pct = passed / len(tests) * 100
    print("=" * 55)
    print(f"📊 النتيجة: {passed}/{len(tests)} ({pct:.0f}%)")
    print("🎉 ممتاز!" if pct >= 85 else "👍 جيد!" if pct >= 70 else "⚠️ يحتاج تحسين")
    print("=" * 55)


if __name__ == "__main__":
    test_parser()
