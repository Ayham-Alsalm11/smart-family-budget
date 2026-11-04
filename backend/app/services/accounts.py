"""Accounts: registration, invites, login with lockout, members and family settings (§5).

Messages are in Arabic here because the API shows them to the user as they are. Steps that two requests could
race on are written as a single conditional UPDATE, or checked after the change inside the same transaction:
an invite is used once, and a family never loses its last active manager.
"""
import math
import re
import secrets
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.auth.passwords import burn_verification_time, hash_password, validate_password, verify_password
from app.db.models import ROLES, Expense, Family, Invite, Member, PendingAction, utcnow
from app.services.currency import parse_amount, parse_rate, validate_currency
from app.services.errors import (
    AuthenticationError, ConflictError, ForbiddenError, LockedError, NotFoundError, ValidationError,
)
from app.services.expenses import get_family

USERNAME = re.compile(r"[a-zA-Z0-9_.]{3,30}")
MAX_NAME = 50
MAX_FAMILY_NAME = 100
INVITE_CODE_LENGTH = 8
INVITE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0, O, 1, I or L
BAD_LOGIN = "اسم المستخدم أو كلمة المرور غير صحيحة."
FAMILY_FIELDS = frozenset({"name", "monthly_income", "base_currency", "usd_rate"})


# ---------- validation ----------

def normalise_username(username: str) -> str:
    if not USERNAME.fullmatch(username):
        raise ValidationError("اسم المستخدم من 3 إلى 30 حرفاً: أحرف إنكليزية وأرقام و_ و. فقط.", ["username"])
    return username.lower()


def _clean_name(name: str, limit: int, field: str, label: str) -> str:
    name = name.strip()
    if not 1 <= len(name) <= limit:
        raise ValidationError(f"{label} من 1 إلى {limit} حرفاً.", [field])
    return name


def _require_manager(ctx: ToolContext, action: str) -> None:
    if not ctx.is_manager:
        raise ForbiddenError(f"only a manager can {action}")


def _require_free_username(session: Session, username: str) -> None:
    if session.scalar(select(Member.id).where(Member.username == username)) is not None:
        raise ConflictError("اسم المستخدم مأخوذ.", ["username"])


def _flush_new_member(session: Session, member: Member) -> None:
    """The unique index decides when two requests register the same username at once."""
    session.add(member)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        raise ConflictError("اسم المستخدم مأخوذ.", ["username"]) from None


# ---------- views ----------

def member_to_dict(member: Member) -> dict:
    return {"id": member.id, "name": member.name, "username": member.username, "role": member.role,
            "is_active": member.is_active, "created_at": member.created_at.isoformat()}


def family_to_dict(family: Family) -> dict:
    return {"id": family.id, "name": family.name, "monthly_income": family.monthly_income,
            "base_currency": family.base_currency, "usd_rate": family.usd_rate}


def invite_to_dict(invite: Invite) -> dict:
    return {"id": invite.id, "code": invite.code, "role": invite.role, "created_by": invite.created_by,
            "expires_at": invite.expires_at.isoformat()}


def profile(session: Session, ctx: ToolContext) -> dict:
    member = session.get(Member, ctx.member_id)
    return {"member": member_to_dict(member), "family": family_to_dict(get_family(session, ctx))}


# ---------- registration and login ----------

def register_family(
    session: Session,
    *,
    family_name: str,
    monthly_income: object,
    base_currency: str = "SYP_NEW",
    usd_rate: object = None,
    name: str,
    username: str,
    password: str,
) -> Member:
    """A new family and its first account, which is a manager (§5)."""
    family = Family(
        name=_clean_name(family_name, MAX_FAMILY_NAME, "family_name", "اسم العائلة"),
        monthly_income=parse_amount(monthly_income, "monthly_income", allow_zero=True),
        base_currency=validate_currency(base_currency),
        usd_rate=None if usd_rate is None else parse_rate(usd_rate),
    )
    display_name = _clean_name(name, MAX_NAME, "name", "الاسم")
    username = normalise_username(username)
    validate_password(password)
    _require_free_username(session, username)

    session.add(family)
    session.flush()
    member = Member(family_id=family.id, name=display_name, username=username,
                    password_hash=hash_password(password), role="manager")
    _flush_new_member(session, member)
    session.commit()
    return member


def _require_open(invite: Invite) -> None:
    if invite.used_by is not None:
        raise ConflictError("كود الدعوة مستعمل.", ["code"])
    if invite.revoked_at is not None:
        raise ConflictError("كود الدعوة ملغى.", ["code"])
    if invite.expires_at <= utcnow():
        raise ConflictError("كود الدعوة منتهي.", ["code"])


def join_with_code(session: Session, *, code: str, name: str, username: str, password: str) -> Member:
    """The family and the role come from the invite, never from the request."""
    invite = session.scalar(select(Invite).where(Invite.code == code.strip().upper()))
    if invite is None:
        raise NotFoundError("كود الدعوة غير صحيح.", ["code"])
    _require_open(invite)
    display_name = _clean_name(name, MAX_NAME, "name", "الاسم")
    username = normalise_username(username)
    validate_password(password)
    _require_free_username(session, username)

    member = Member(family_id=invite.family_id, name=display_name, username=username,
                    password_hash=hash_password(password), role=invite.role)
    _flush_new_member(session, member)
    claimed = session.execute(
        update(Invite).where(Invite.id == invite.id, Invite.used_by.is_(None), Invite.revoked_at.is_(None))
        .values(used_by=member.id, used_at=utcnow())
    ).rowcount
    if not claimed:
        session.rollback()
        raise ConflictError("كود الدعوة مستعمل.", ["code"])
    session.commit()
    return member


def _record_failed_login(session: Session, member: Member, max_failures: int, lock_minutes: int) -> None:
    member.failed_logins = Member.failed_logins + 1  # in SQL, so that parallel guesses are all counted
    session.flush()
    session.refresh(member)
    if member.failed_logins >= max_failures:
        member.locked_until = utcnow() + timedelta(minutes=lock_minutes)
    session.commit()


def login(session: Session, username: str, password: str, *, max_failures: int = 5, lock_minutes: int = 5) -> Member:
    """The same message for an unknown username, a wrong password and a stopped account (§5)."""
    member = session.scalar(select(Member).where(Member.username == username.strip().lower()))
    if member is None or not member.is_active:
        burn_verification_time(password)
        raise AuthenticationError(BAD_LOGIN)

    now = utcnow()
    if member.locked_until is not None:
        if member.locked_until > now:
            seconds = (member.locked_until - now).total_seconds()
            raise LockedError(f"الدخول مقفل مؤقتاً بعد عدة محاولات فاشلة. حاول بعد {math.ceil(seconds / 60)} دقيقة.",
                              math.ceil(seconds))
        member.failed_logins, member.locked_until = 0, None  # the lock has passed: a fresh count
        session.commit()

    if not verify_password(password, member.password_hash):
        _record_failed_login(session, member, max_failures, lock_minutes)
        raise AuthenticationError(BAD_LOGIN)
    member.failed_logins, member.locked_until = 0, None
    session.commit()
    return member


# ---------- invites ----------

def _new_code(session: Session) -> str:
    while True:
        code = "".join(secrets.choice(INVITE_ALPHABET) for _ in range(INVITE_CODE_LENGTH))
        if session.scalar(select(Invite.id).where(Invite.code == code)) is None:
            return code


def create_invite(session: Session, ctx: ToolContext, role: str = "member", days: int = 7) -> dict:
    _require_manager(ctx, "create invites")
    if role not in ROLES:
        raise ValidationError(f"الدور واحد من: {', '.join(ROLES)}.", ["role"])
    invite = Invite(family_id=ctx.family_id, code=_new_code(session), role=role, created_by=ctx.member_id,
                    expires_at=utcnow() + timedelta(days=days))
    session.add(invite)
    session.commit()
    return invite_to_dict(invite)


def list_invites(session: Session, ctx: ToolContext) -> dict:
    """Open invites only: not used, not revoked and not expired."""
    _require_manager(ctx, "see invites")
    rows = session.scalars(select(Invite).where(
        Invite.family_id == ctx.family_id, Invite.used_by.is_(None), Invite.revoked_at.is_(None),
        Invite.expires_at > utcnow(),
    ).order_by(Invite.id.desc())).all()
    return {"invites": [invite_to_dict(i) for i in rows]}


def revoke_invite(session: Session, ctx: ToolContext, invite_id: int) -> dict:
    _require_manager(ctx, "revoke invites")
    invite = session.get(Invite, invite_id)
    if invite is None or invite.family_id != ctx.family_id:
        raise NotFoundError(f"invite #{invite_id} not found")
    _require_open(invite)
    revoked = session.execute(
        update(Invite).where(Invite.id == invite_id, Invite.used_by.is_(None), Invite.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    ).rowcount
    if not revoked:
        session.rollback()
        raise ConflictError("كود الدعوة مستعمل.", ["code"])
    session.commit()
    return {"id": invite_id, "revoked": True}


# ---------- members ----------

def list_members(session: Session, ctx: ToolContext) -> dict:
    _require_manager(ctx, "see the members")
    rows = session.scalars(select(Member).where(Member.family_id == ctx.family_id).order_by(Member.id)).all()
    return {"members": [member_to_dict(m) for m in rows]}


def _active_member_of_family(session: Session, ctx: ToolContext, member_id: int) -> Member:
    member = session.get(Member, member_id)
    if member is None or member.family_id != ctx.family_id:
        raise NotFoundError(f"member #{member_id} not found")
    if not member.is_active:
        raise ConflictError("هذا الحساب موقوف.")
    return member


def _keep_one_manager(session: Session, family_id: int) -> None:
    """Runs after the change, in the same transaction, so two managers cannot remove each other at once."""
    session.flush()
    managers = session.scalar(select(func.count()).select_from(Member).where(
        Member.family_id == family_id, Member.role == "manager", Member.is_active.is_(True)))
    if not managers:
        session.rollback()
        raise ConflictError("لا يمكن إزالة آخر مسؤول نشط في العائلة.")


def deactivate_member(session: Session, ctx: ToolContext, member_id: int) -> dict:
    """No row is deleted: the expenses stay in every total under the member's name (§5)."""
    _require_manager(ctx, "stop members")
    member = _active_member_of_family(session, ctx, member_id)
    was_manager = member.role == "manager"
    member.is_active = False
    if was_manager:
        _keep_one_manager(session, ctx.family_id)
    session.execute(update(PendingAction).where(
        PendingAction.member_id == member_id, PendingAction.status == "pending").values(status="cancelled"))
    if was_manager:
        now = utcnow()
        session.execute(update(Invite).where(
            Invite.created_by == member_id, Invite.used_by.is_(None), Invite.revoked_at.is_(None),
            Invite.expires_at > now).values(revoked_at=now))
    session.commit()
    return member_to_dict(member)


def change_role(session: Session, ctx: ToolContext, member_id: int, role: str) -> dict:
    _require_manager(ctx, "change roles")
    if role not in ROLES:
        raise ValidationError(f"الدور واحد من: {', '.join(ROLES)}.", ["role"])
    member = _active_member_of_family(session, ctx, member_id)
    demoted = member.role == "manager" and role != "manager"
    member.role = role
    if demoted:
        _keep_one_manager(session, ctx.family_id)
    session.commit()
    return member_to_dict(member)


# ---------- family ----------

def update_family(session: Session, ctx: ToolContext, changes: dict[str, Any]) -> dict:
    """Base currency: only while the family has no expenses, because `amount_base` is computed in it.
    The USD rate: freely, because each expense keeps its own `rate_to_base`."""
    _require_manager(ctx, "change the family settings")
    if not changes:
        raise ValidationError("no changes given")
    unknown = sorted(set(changes) - FAMILY_FIELDS)
    if unknown:
        raise ValidationError(f"these fields cannot be changed: {', '.join(unknown)}", unknown)
    family = get_family(session, ctx)

    values: dict[str, Any] = {}
    if "name" in changes:
        values["name"] = _clean_name(changes["name"] or "", MAX_FAMILY_NAME, "name", "اسم العائلة")
    if "monthly_income" in changes:
        values["monthly_income"] = parse_amount(changes["monthly_income"], "monthly_income", allow_zero=True)
    if "usd_rate" in changes:
        values["usd_rate"] = parse_rate(changes["usd_rate"])
    if "base_currency" in changes:
        currency = validate_currency(changes["base_currency"])
        if currency != family.base_currency:
            if session.scalar(select(Expense.id).where(Expense.family_id == family.id).limit(1)) is not None:
                raise ConflictError("لا يمكن تغيير عملة الأساس لأن للعائلة مصاريف مسجّلة.", ["base_currency"])
            values["base_currency"] = currency

    for field, value in values.items():
        setattr(family, field, value)
    session.commit()
    return family_to_dict(family)
