"""
اختبار BGE-M3 Embedding
تأكد إن Ollama مشغّل: ollama serve
وإن BGE-M3 محمّل: ollama pull bge-m3
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag.embedding import EmbeddingEngine


def test_embedding():
    print("=" * 55)
    print("🔌 اختبار BGE-M3 Embedding...")
    print("=" * 55)

    engine = EmbeddingEngine()

    # اختبار الاتصال
    if not engine.test_connection():
        print("\n⚠️ تأكد من:")
        print("  1. Ollama مشغّل: ollama serve")
        print("  2. BGE-M3 محمّل: ollama pull bge-m3")
        return

    # اختبار إضافة مصاريف
    print("\n📝 إضافة مصاريف تجريبية...")
    test_expenses = [
        (1, "دفعت بنزين للسيارة", {"amount": "5000", "category": "بنزين ومواصلات", "date": "2026-03-27", "description": "بنزين"}),
        (2, "اشتريت خضرا من السوق", {"amount": "3000", "category": "أكل وشرب", "date": "2026-03-27", "description": "خضار"}),
        (3, "دفعت فاتورة الكهربا", {"amount": "15000", "category": "فواتير", "date": "2026-03-27", "description": "كهربا"}),
        (4, "رحت عالدكتور", {"amount": "20000", "category": "صحة", "date": "2026-03-27", "description": "دكتور"}),
        (5, "غدا من المطعم", {"amount": "8000", "category": "أكل وشرب", "date": "2026-03-27", "description": "مطعم"}),
    ]

    for eid, text, meta in test_expenses:
        engine.add_expense(eid, text, meta)
        print(f"  ✅ {text}")

    # اختبار البحث
    print("\n🔍 اختبار البحث الدلالي...")
    queries = ["كم صرفت أكل؟", "مصاريف السيارة", "فواتير"]

    for q in queries:
        results = engine.search_similar(q, n_results=2)
        print(f"\n  سؤال: \"{q}\"")
        for r in results:
            print(f"    → {r['text']} ({r['metadata'].get('category', '?')})")

    # إحصائيات
    stats = engine.get_stats()
    print(f"\n📊 ChromaDB: {stats['total_documents']} مصروف مخزّن")
    print("\n✅ BGE-M3 + ChromaDB يعملان بنجاح!")


if __name__ == "__main__":
    test_embedding()
