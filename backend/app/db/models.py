"""Database schema (ARCHITECTURE.md §4). The `events` table is Priority 4 and not created yet."""
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

CURRENCIES = ("SYP_NEW", "SYP_OLD", "USD")
ROLES = ("manager", "member")
GOAL_STATUSES = ("active", "done", "cancelled")
ACTION_STATUSES = ("pending", "confirmed", "cancelled", "expired")
MESSAGE_ROLES = ("user", "assistant")
TOOL_STATUSES = (
    "ok", "error", "invalid_args", "forbidden", "confirmation_required", "not_found", "ungrounded",
)

Money = Numeric(14, 2)
Rate = Numeric(20, 10)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _one_of(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC timestamps. SQLite has no time zones, so values are stored as naive UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime: use a timezone-aware UTC value")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        return None if value is None else value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class Family(Base):
    __tablename__ = "families"
    __table_args__ = (
        CheckConstraint(_one_of("base_currency", CURRENCIES), name="ck_families_base_currency"),
        CheckConstraint("usd_rate IS NULL OR usd_rate > 0", name="ck_families_usd_rate"),
        CheckConstraint("monthly_income >= 0", name="ck_families_income"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    monthly_income: Mapped[Decimal] = mapped_column(Money, default=Decimal("0"))
    base_currency: Mapped[str] = mapped_column(String(8), default="SYP_NEW")
    usd_rate: Mapped[Decimal | None] = mapped_column(Rate)  # units of new SYP per 1 USD
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Member(Base):
    __tablename__ = "members"
    __table_args__ = (CheckConstraint(_one_of("role", ROLES), name="ck_members_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(10))
    is_active: Mapped[bool] = mapped_column(default=True)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)  # consecutive; reset by a successful login
    locked_until: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Invite(Base):
    __tablename__ = "invites"
    __table_args__ = (CheckConstraint(_one_of("role", ROLES), name="ck_invites_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"), index=True)
    code: Mapped[str] = mapped_column(String(8), unique=True)
    role: Mapped[str] = mapped_column(String(10))
    created_by: Mapped[int] = mapped_column(ForeignKey("members.id"))
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    used_by: Mapped[int | None] = mapped_column(ForeignKey("members.id"))
    used_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name_ar: Mapped[str] = mapped_column(String(50), unique=True)
    name_en: Mapped[str] = mapped_column(String(50))
    budget_percent: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    icon: Mapped[str] = mapped_column(String(16))


class SavingsGoal(Base):
    __tablename__ = "savings_goals"
    __table_args__ = (
        CheckConstraint(_one_of("status", GOAL_STATUSES), name="ck_savings_goals_status"),
        CheckConstraint("target_amount > 0", name="ck_savings_goals_target"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    target_amount: Mapped[Decimal] = mapped_column(Money)
    deadline: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(10), default="active")
    created_by: Mapped[int] = mapped_column(ForeignKey("members.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Expense(Base):
    __tablename__ = "expenses"
    __table_args__ = (
        CheckConstraint(_one_of("currency", CURRENCIES), name="ck_expenses_currency"),
        CheckConstraint("amount > 0", name="ck_expenses_amount"),
        CheckConstraint("amount_base >= 0", name="ck_expenses_amount_base"),
        CheckConstraint("rate_to_base > 0", name="ck_expenses_rate"),
        Index("ix_expenses_family_date", "family_id", "expense_date"),
        Index("ix_expenses_member", "member_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"))
    member_id: Mapped[int] = mapped_column(ForeignKey("members.id"))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    amount: Mapped[Decimal] = mapped_column(Money)
    currency: Mapped[str] = mapped_column(String(8))
    amount_base: Mapped[Decimal] = mapped_column(Money)
    rate_to_base: Mapped[Decimal] = mapped_column(Rate)
    description: Mapped[str] = mapped_column(String(255), default="")
    source_text: Mapped[str] = mapped_column(Text, default="")
    expense_date: Mapped[date] = mapped_column(Date)
    goal_id: Mapped[int | None] = mapped_column(ForeignKey("savings_goals.id"))
    created_by_agent: Mapped[bool] = mapped_column(default=False)
    embedded_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    category: Mapped[Category] = relationship()


class Budget(Base):
    __tablename__ = "budgets"
    __table_args__ = (
        CheckConstraint("monthly_limit >= 0", name="ck_budgets_limit"),
        CheckConstraint("month IS NULL OR month LIKE '____-__'", name="ck_budgets_month"),
        UniqueConstraint("family_id", "category_id", "month", name="uq_budgets_family_category_month"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"))
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    monthly_limit: Mapped[Decimal] = mapped_column(Money)
    month: Mapped[str | None] = mapped_column(String(7))  # YYYY-MM, or NULL for a recurring default
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(_one_of("role", MESSAGE_ROLES), name="ck_messages_role"),
        Index("ix_messages_member_created", "member_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"))
    member_id: Mapped[int] = mapped_column(ForeignKey("members.id"))
    role: Mapped[str] = mapped_column(String(10))
    content: Mapped[str] = mapped_column(Text)
    tool_summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class PendingAction(Base):
    __tablename__ = "pending_actions"
    __table_args__ = (CheckConstraint(_one_of("status", ACTION_STATUSES), name="ck_pending_actions_status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"))
    member_id: Mapped[int] = mapped_column(ForeignKey("members.id"), index=True)
    message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"))
    tool_name: Mapped[str] = mapped_column(String(50))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(10), default="pending")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"), index=True)
    member_id: Mapped[int] = mapped_column(ForeignKey("members.id"))
    message: Mapped[str] = mapped_column(Text)
    wrong_call: Mapped[dict[str, Any]] = mapped_column(JSON)
    corrected_call: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ToolLog(Base):
    __tablename__ = "tool_log"
    __table_args__ = (CheckConstraint(_one_of("status", TOOL_STATUSES), name="ck_tool_log_status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"), index=True)
    family_id: Mapped[int] = mapped_column(ForeignKey("families.id"))
    member_id: Mapped[int] = mapped_column(ForeignKey("members.id"))
    step: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(50))
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(25))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class LlmCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"), index=True)
    model: Mapped[str] = mapped_column(String(100))
    step: Mapped[int] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    http_status: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)  # part of completion_tokens (§6.3)
    cached_tokens: Mapped[int | None] = mapped_column(Integer)  # part of prompt_tokens (§6.3)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
