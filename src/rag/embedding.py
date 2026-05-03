"""
Embedding Module — BGE-M3 عبر Ollama + ChromaDB
يحوّل النصوص لـ vectors ويخزّنها ويبحث فيها
"""
import requests
import chromadb
import os
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL, CHROMA_PERSIST_DIR


class EmbeddingEngine:
    """محرك الـ Embedding — BGE-M3 + ChromaDB"""

    def __init__(self):
        """تهيئة الاتصال بـ Ollama و ChromaDB"""
        self.ollama_url = f"{OLLAMA_BASE_URL}/api/embed"
        self.model = EMBEDDING_MODEL

        # إنشاء مجلد ChromaDB
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # الاتصال بـ ChromaDB
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # إنشاء collection للمصروفات
        self.collection = self.client.get_or_create_collection(
            name="expenses",
            metadata={"description": "مصروفات المستخدم"}
        )

    def get_embedding(self, text: str) -> list:
        """
        تحويل نص إلى vector باستخدام BGE-M3 عبر Ollama

        Args:
            text: النص المراد تحويله
        Returns:
            list: vector (قائمة أرقام)
        """
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.model, "input": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
        except requests.exceptions.ConnectionError:
            print("❌ Ollama غير مشغّل! شغّله بأمر: ollama serve")
            return None
        except Exception as e:
            print(f"❌ خطأ بالـ Embedding: {e}")
            return None

    def add_expense(self, expense_id: int, text: str, metadata: dict):
        """
        إضافة مصروف لـ ChromaDB مع الـ embedding

        Args:
            expense_id: رقم المصروف من قاعدة البيانات
            text: النص الأصلي للمصروف
            metadata: بيانات إضافية (المبلغ، الفئة، التاريخ)
        """
        embedding = self.get_embedding(text)
        if embedding is None:
            print("⚠️ لم يتم إضافة الـ embedding — Ollama غير متاح")
            return

        # تحويل كل القيم في metadata لـ string
        safe_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                safe_metadata[key] = str(value)

        self.collection.upsert(
            ids=[str(expense_id)],
            embeddings=[embedding],
            documents=[text],
            metadatas=[safe_metadata]
        )

    def search_similar(self, query: str, n_results: int = 5) -> list:
        """
        البحث عن مصاريف مشابهة

        Args:
            query: سؤال أو نص للبحث
            n_results: عدد النتائج
        Returns:
            list: قائمة بالنتائج المشابهة
        """
        embedding = self.get_embedding(query)
        if embedding is None:
            return []

        # التأكد إن في بيانات
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self.collection.count())
        )

        # تنسيق النتائج
        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "id": results["ids"][0][i],
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })
        return formatted

    def get_context_for_query(self, query: str, n_results: int = 10) -> str:
        """
        استرجاع سياق للـ RAG — يُرسل مع السؤال للـ LLM

        Args:
            query: سؤال المستخدم
        Returns:
            str: نص السياق
        """
        results = self.search_similar(query, n_results)
        if not results:
            return "لا توجد مصاريف مسجلة بعد."

        context = "=== المصاريف المسترجعة ===\n"
        for r in results:
            meta = r["metadata"]
            context += (
                f"  - {meta.get('description', '')} | "
                f"{meta.get('amount', '?')} ل.س | "
                f"{meta.get('category', '?')} | "
                f"{meta.get('date', '?')}\n"
            )
        return context

    def test_connection(self) -> bool:
        """اختبار الاتصال بـ Ollama"""
        try:
            embedding = self.get_embedding("اختبار")
            if embedding:
                print(f"✅ BGE-M3 يعمل! حجم الـ vector: {len(embedding)}")
                return True
            return False
        except Exception as e:
            print(f"❌ فشل الاتصال بـ Ollama: {e}")
            return False

    def get_stats(self) -> dict:
        """إحصائيات ChromaDB"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }
