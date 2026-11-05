"""The 12 expense categories (§3.1, §4), copied from GP1 (baseline/config.py; icons from baseline/src/database/seed.py).

Copied rather than imported, so that the baseline stays frozen (D12).
"""

CATEGORIES = [
    {
        "name_ar": "أكل وشرب", "name_en": "Food & Drinks", "budget_percent": 20, "icon": "🍽️",
        "keywords": ["مطعم", "سوبرماركت", "خضرا", "لحمة", "فول", "خبز", "حليب",
                     "فلافل", "شاورما", "أكل", "طعام", "غدا", "عشا", "فطور",
                     "سوق", "بقالة", "دكان", "جبنة", "رز", "زيت", "سكر",
                     "فواكه", "عصير", "ماي شرب", "شاي", "قهوة"],
    },
    {
        "name_ar": "بنزين ومواصلات", "name_en": "Fuel & Transport", "budget_percent": 15, "icon": "⛽",
        "keywords": ["بنزين", "تكسي", "باص", "سرفيس", "مازوت", "غاز",
                     "مواصلات", "نقل", "سيارة", "بنشر", "زيت سيارة"],
    },
    {
        "name_ar": "فواتير", "name_en": "Bills & Utilities", "budget_percent": 10, "icon": "💡",
        "keywords": ["كهربا", "كهرباء", "ماي", "مياه", "نت", "إنترنت",
                     "موبايل", "هاتف", "فاتورة", "اشتراك"],
    },
    {
        "name_ar": "إيجار", "name_en": "Rent", "budget_percent": 25, "icon": "🏠",
        "keywords": ["إيجار", "أجار", "إجار", "بيت", "شقة"],
    },
    {
        "name_ar": "صحة", "name_en": "Health", "budget_percent": 5, "icon": "🏥",
        "keywords": ["دكتور", "طبيب", "صيدلية", "دوا", "دواء", "مشفى",
                     "مستشفى", "تحاليل", "أشعة", "عملية", "أسنان"],
    },
    {
        "name_ar": "تعليم", "name_en": "Education", "budget_percent": 5, "icon": "📚",
        "keywords": ["مدرسة", "جامعة", "كورس", "كتب", "كتاب", "دورة",
                     "تدريب", "قرطاسية", "أقساط", "رسوم"],
    },
    {
        "name_ar": "ملابس", "name_en": "Clothing", "budget_percent": 5, "icon": "👕",
        "keywords": ["ثياب", "جاكيت", "بنطلون", "قميص", "حذاء", "جزمة",
                     "ملابس", "فستان", "عباية"],
    },
    {
        "name_ar": "ترفيه", "name_en": "Entertainment", "budget_percent": 5, "icon": "🎬",
        "keywords": ["سينما", "نزهة", "كافيه", "مقهى", "رحلة", "سفر",
                     "لعبة", "ألعاب", "نادي", "سبورت", "مسبح"],
    },
    {
        "name_ar": "ادخار", "name_en": "Savings", "budget_percent": 5, "icon": "💰",
        "keywords": ["ادخار", "توفير", "حطيت جنب", "وفرت"],
    },
    {
        "name_ar": "هدايا ومناسبات", "name_en": "Gifts & Events", "budget_percent": 2, "icon": "🎁",
        "keywords": ["هدية", "عزيمة", "مناسبة", "عرس", "خطوبة", "عيد"],
    },
    {
        "name_ar": "صيانة", "name_en": "Maintenance", "budget_percent": 2, "icon": "🔧",
        "keywords": ["تصليح", "صيانة", "سباك", "كهربائي", "نجار",
                     "دهان", "تمديدات"],
    },
    {
        "name_ar": "متفرقات", "name_en": "Miscellaneous", "budget_percent": 1, "icon": "📦",
        "keywords": [],
    },
]

CATEGORY_NAMES = [c["name_ar"] for c in CATEGORIES]
SAVINGS_CATEGORY = "ادخار"
