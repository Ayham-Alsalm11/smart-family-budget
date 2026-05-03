"""
Database Models — جداول قاعدة البيانات
"""
from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date,
    DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL

Base = declarative_base()


class Family(Base):
    """العائلة"""
    __tablename__ = "families"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    monthly_income = Column(Float, default=0)
    currency = Column(String(10), default="SYP")
    created_at = Column(DateTime, default=datetime.utcnow)
    expenses = relationship("Expense", back_populates="family", cascade="all, delete-orphan")
    budgets = relationship("Budget", back_populates="family", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Family(id={self.id}, name='{self.name}', income={self.monthly_income})>"


class Category(Base):
    """فئات المصروفات — 12 فئة"""
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name_ar = Column(String(50), nullable=False, unique=True)
    name_en = Column(String(50), nullable=False)
    budget_percent = Column(Float, default=0)
    keywords = Column(Text, default="")
    icon = Column(String(10), default="💰")
    expenses = relationship("Expense", back_populates="category")
    budgets = relationship("Budget", back_populates="category")

    def __repr__(self):
        return f"<Category(id={self.id}, name='{self.name_ar}')>"

    def get_keywords_list(self):
        if not self.keywords:
            return []
        return [kw.strip() for kw in self.keywords.split(",") if kw.strip()]


class Expense(Base):
    """سجل المصروفات"""
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True, autoincrement=True)
    family_id = Column(Integer, ForeignKey("families.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(String(255), default="")
    source_text = Column(Text, default="")
    expense_date = Column(Date, default=date.today)
    created_at = Column(DateTime, default=datetime.utcnow)
    family = relationship("Family", back_populates="expenses")
    category = relationship("Category", back_populates="expenses")

    def __repr__(self):
        return f"<Expense(id={self.id}, amount={self.amount})>"


class Budget(Base):
    """الميزانية الشهرية"""
    __tablename__ = "budgets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    family_id = Column(Integer, ForeignKey("families.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    monthly_limit = Column(Float, nullable=False)
    month = Column(String(7), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    family = relationship("Family", back_populates="budgets")
    category = relationship("Category", back_populates="budgets")


def get_engine():
    os.makedirs(os.path.dirname(DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)
    return create_engine(DATABASE_URL, echo=False)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("✅ تم إنشاء قاعدة البيانات بنجاح!")
    return engine


if __name__ == "__main__":
    init_db()
