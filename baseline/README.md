# 💰 ميزان — مساعد الميزانية العائلي الذكي
# Smart Family Budget Assistant — LLM + RAG

نظام ذكي لإدارة المصروفات العائلية باللغة العربية واللهجة السورية.

## 🏗️ البنية التقنية
```
Groq API (سحابة):          Ollama (محلي):
└── Llama 3.3 70B           └── BGE-M3 (embedding)
    (فهم + توليد)               (تحويل نص → vector)
         ↕                           ↕
    ┌─────────────────────────────────────┐
    │         Chat Engine                 │
    │    (يربط كل المكونات)              │
    ├──────────┬──────────┬───────────────┤
    │ SQLite   │ ChromaDB │  Streamlit    │
    │ (تخزين)  │ (vectors)│  (واجهة)     │
    └──────────┴──────────┴───────────────┘
```

## 🚀 التشغيل

### 1. إعداد البيئة
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. إعداد قاعدة البيانات
```bash
python -m src.database.seed
```

### 3. تثبيت Ollama + BGE-M3
```bash
# حمّل Ollama: https://ollama.com/download
ollama pull bge-m3
ollama serve
```

### 4. إعداد Groq API
```bash
# سجّل: https://console.groq.com/keys
# PowerShell:
$env:GROQ_API_KEY="gsk_xxx"
# CMD:
set GROQ_API_KEY=gsk_xxx
```

### 5. تشغيل الاختبارات
```bash
python -m tests.test_embedding   # اختبار BGE-M3
python -m tests.test_parser      # اختبار LLM Parser
python -m tests.demo_flow        # محادثة تفاعلية
```

### 6. تشغيل التطبيق
```bash
streamlit run app.py
```

## 📁 هيكل المشروع
```
smart-family-budget/
├── app.py                          # واجهة Streamlit
├── config.py                       # إعدادات + 12 فئة
├── src/
│   ├── database/
│   │   ├── models.py               # جداول DB
│   │   ├── crud.py                 # عمليات CRUD
│   │   └── seed.py                 # بيانات أولية
│   ├── llm/
│   │   ├── prompts.py              # System Prompt + قواعد مالية
│   │   └── parser.py               # Groq API parser
│   ├── rag/
│   │   ├── embedding.py            # BGE-M3 + ChromaDB
│   │   └── retriever.py            # RAG retriever (SQL + embedding)
│   ├── analysis/
│   │   └── spending.py             # تحليل الإنفاق
│   └── chat/
│       └── engine.py               # محرك المحادثة
└── tests/
    ├── test_parser.py              # اختبار LLM
    ├── test_embedding.py           # اختبار BGE-M3
    └── demo_flow.py                # عرض تجريبي
```
