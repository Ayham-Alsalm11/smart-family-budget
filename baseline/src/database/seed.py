"""
Seed Data — تعبئة البيانات الأولية
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.models import init_db, get_session, Category, Family
from config import CATEGORIES

ICONS = {
    "أكل وشرب": "🍽️", "بنزين ومواصلات": "⛽", "فواتير": "💡",
    "إيجار": "🏠", "صحة": "🏥", "تعليم": "📚",
    "ملابس": "👕", "ترفيه": "🎬", "ادخار": "💰",
    "هدايا ومناسبات": "🎁", "صيانة": "🔧", "متفرقات": "📦",
}


def seed_categories():
    session = get_session()
    if session.query(Category).count() > 0:
        print(f"⚠️ الفئات موجودة مسبقاً ({session.query(Category).count()} فئة)")
        session.close()
        return
    for name_ar, info in CATEGORIES.items():
        session.add(Category(
            name_ar=name_ar, name_en=info["en"],
            budget_percent=info["budget_percent"],
            keywords=",".join(info["keywords"]),
            icon=ICONS.get(name_ar, "💰")
        ))
    session.commit()
    print(f"✅ تم إنشاء {len(CATEGORIES)} فئة بنجاح!")
    session.close()


def seed_sample_family():
    session = get_session()
    if session.query(Family).count() > 0:
        print(f"⚠️ يوجد عائلات مسبقاً ({session.query(Family).count()})")
        session.close()
        return
    family = Family(name="عائلة السالم", monthly_income=500000, currency="SYP")
    session.add(family)
    session.commit()
    print(f"✅ تم إنشاء عائلة '{family.name}' بدخل {family.monthly_income:,.0f} ل.س")
    session.close()


def seed_all():
    print("=" * 50)
    print("🔧 إعداد قاعدة البيانات...")
    print("=" * 50)
    init_db()
    seed_categories()
    seed_sample_family()
    print("\n" + "=" * 50)
    print("📊 ملخص قاعدة البيانات:")
    print("=" * 50)
    session = get_session()
    print(f"   العائلات: {session.query(Family).count()}")
    print(f"   الفئات:   {session.query(Category).count()}")
    print(f"\n   📋 الفئات المتاحة:")
    for cat in session.query(Category).all():
        kw = len(cat.get_keywords_list())
        print(f"      {cat.icon} {cat.name_ar} ({cat.name_en}) — {cat.budget_percent}% | {kw} كلمة مفتاحية")
    session.close()
    print("\n✅ قاعدة البيانات جاهزة!")


if __name__ == "__main__":
    seed_all()
