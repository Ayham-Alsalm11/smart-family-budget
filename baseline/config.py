"""
Smart Family Budget Assistant - Configuration
"""
import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "data", "budget.db")

# === Database ===
DATABASE_URL = f"sqlite:///{DB_PATH}"

# === LLM (Groq API) ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_API_KEY_HERE")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# === Embedding (Ollama + BGE-M3) ===
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"

# === ChromaDB ===
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# === Categories (12 فئة) ===
CATEGORIES = {
    "أكل وشرب": {
        "en": "Food & Drinks",
        "budget_percent": 20,
        "keywords": ["مطعم", "سوبرماركت", "خضرا", "لحمة", "فول", "خبز", "حليب",
                     "فلافل", "شاورما", "أكل", "طعام", "غدا", "عشا", "فطور",
                     "سوق", "بقالة", "دكان", "جبنة", "رز", "زيت", "سكر",
                     "فواكه", "عصير", "ماي شرب", "شاي", "قهوة"]
    },
    "بنزين ومواصلات": {
        "en": "Fuel & Transport",
        "budget_percent": 15,
        "keywords": ["بنزين", "تكسي", "باص", "سرفيس", "مازوت", "غاز",
                     "مواصلات", "نقل", "سيارة", "بنشر", "زيت سيارة"]
    },
    "فواتير": {
        "en": "Bills & Utilities",
        "budget_percent": 10,
        "keywords": ["كهربا", "كهرباء", "ماي", "مياه", "نت", "إنترنت",
                     "موبايل", "هاتف", "فاتورة", "اشتراك"]
    },
    "إيجار": {
        "en": "Rent",
        "budget_percent": 25,
        "keywords": ["إيجار", "أجار", "إجار", "بيت", "شقة"]
    },
    "صحة": {
        "en": "Health",
        "budget_percent": 5,
        "keywords": ["دكتور", "طبيب", "صيدلية", "دوا", "دواء", "مشفى",
                     "مستشفى", "تحاليل", "أشعة", "عملية", "أسنان"]
    },
    "تعليم": {
        "en": "Education",
        "budget_percent": 5,
        "keywords": ["مدرسة", "جامعة", "كورس", "كتب", "كتاب", "دورة",
                     "تدريب", "قرطاسية", "أقساط", "رسوم"]
    },
    "ملابس": {
        "en": "Clothing",
        "budget_percent": 5,
        "keywords": ["ثياب", "جاكيت", "بنطلون", "قميص", "حذاء", "جزمة",
                     "ملابس", "فستان", "عباية"]
    },
    "ترفيه": {
        "en": "Entertainment",
        "budget_percent": 5,
        "keywords": ["سينما", "نزهة", "كافيه", "مقهى", "رحلة", "سفر",
                     "لعبة", "ألعاب", "نادي", "سبورت", "مسبح"]
    },
    "ادخار": {
        "en": "Savings",
        "budget_percent": 5,
        "keywords": ["ادخار", "توفير", "حطيت جنب", "وفرت"]
    },
    "هدايا ومناسبات": {
        "en": "Gifts & Events",
        "budget_percent": 2,
        "keywords": ["هدية", "عزيمة", "مناسبة", "عرس", "خطوبة", "عيد"]
    },
    "صيانة": {
        "en": "Maintenance",
        "budget_percent": 2,
        "keywords": ["تصليح", "صيانة", "سباك", "كهربائي", "نجار",
                     "دهان", "تمديدات"]
    },
    "متفرقات": {
        "en": "Miscellaneous",
        "budget_percent": 1,
        "keywords": []
    }
}
