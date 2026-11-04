"""Confirmation in chat (§6.5): the model can never run an edit or a deletion.

The server writes the preview, and a short yes or no from the user runs or cancels the action without any model
call. The rules are fixed and live only here. Anything they do not recognise is an ordinary turn, and the action
stays pending until it expires, so a doubtful message never changes data.
"""
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.agent.prompts import CURRENCY_NAMES
from app.db.models import Message, PendingAction
from app.services.errors import ConflictError, ForbiddenError, NotFoundError, ServiceError
from app.services.pending import CONFIRMATION_MINUTES, cancel_action, execute_action
from app.tools.dispatch import dumps

MAX_WORDS = 4
# All in the form `normalise` produces: no hamza and no diacritics.
YES_WORDS = {"اي", "ايه", "ايوه", "ايوا", "اه", "نعم", "اكيد", "تمام", "موافق", "موافقه", "اكد", "اكدها", "اكده",
             "ماشي", "اوكي", "اوك", "ok", "okay", "yes", "yep"}
NO_WORDS = {"لا", "لاء", "الغي", "الغيها", "الغيه", "الغاء", "بلاش", "خليه", "خليها", "كنسل", "no", "nope", "cancel"}
NEGATIONS = {"مو", "مش", "ما", "مانو", "مني", "ماني", "مابدي", "غير", "not"}  # "مو موافق" must never read as a yes

_HAMZA = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ي"})
_PUNCTUATION = re.compile(r"[^\w\s]|_")

REPLY_CANCELLED = "تمام، ما غيّرت شي."
REPLY_EXPIRED = f"انتهت مهلة التأكيد ({CONFIRMATION_MINUTES} دقائق)، وما غيّرت شي. إذا لسا بدك، اطلبها من جديد."
REPLY_NOT_FOUND = "هالمصروف ما عاد موجود، فما في شي لنفّذه."
REPLY_ALREADY_DONE = "هالطلب منفَّذ من قبل، وما عاد في شي لنغيّره."
REPLY_FORBIDDEN = "ما عاد عندك صلاحية لهالتغيير، وما غيّرت شي."
REPLY_FAILED = "ما قدرت نفّذ هالتغيير، وما غيّرت شي."


def normalise(text: str) -> str:
    """Without diacritics, punctuation and hamza forms, in lower case, with single spaces."""
    text = "".join(c for c in unicodedata.normalize("NFC", text)
                   if unicodedata.category(c) not in ("Mn", "Cf") and c != "ـ")
    return " ".join(_PUNCTUATION.sub(" ", text.translate(_HAMZA).lower()).split())


def classify(message: str) -> str | None:
    """'yes', 'no', or None for anything else: too long, mixed, with a number, or negated."""
    words = normalise(message).split()
    if not words or len(words) > MAX_WORDS or any(c.isdigit() for word in words for c in word):
        return None
    unique = set(words)
    if unique & NEGATIONS:
        return None
    yes, no = bool(unique & YES_WORDS), bool(unique & NO_WORDS)
    if yes == no:
        return None
    return "yes" if yes else "no"


# ---------- text written by the server ----------

def _money(amount: Any, currency: str) -> str:
    return f"{f'{float(amount):,.2f}'.removesuffix('.00')} {CURRENCY_NAMES[currency]}"


def _line(expense: dict) -> str:
    parts = [_money(expense["amount"], expense["currency"]), expense["category"], expense["description"],
             expense["expense_date"]]
    return " · ".join(part for part in parts if part)


def preview_text(pending: dict) -> str:
    """The question for an edit or a deletion, from the row in the database. The model has no part in it."""
    before = pending["expense"]
    ask = f"رد بـ «اي» للتأكيد أو «لا» للإلغاء، خلال {CONFIRMATION_MINUTES} دقائق."
    if pending["action"] == "delete_expense":
        return f"بدي احذف هالمصروف:\n{_line(before)}\n{ask}"
    changes = pending["changes"]
    after = {**before, **{key: value for key, value in changes.items() if key != "date"}}
    if "date" in changes:
        after["expense_date"] = changes["date"]
    return f"بدي عدّل هالمصروف:\nقبل: {_line(before)}\nبعد: {_line(after)}\n{ask}"


# ---------- answering a pending action ----------

@dataclass
class ConfirmationResult:
    reply: str
    ended_by: str  # confirmed, cancelled, confirm_error
    tool: str  # confirm_action, cancel_action
    status: str  # as in tool_log
    arguments: dict
    result: dict
    latency_ms: int


def waiting_for_an_answer(session: Session, ctx: ToolContext) -> bool:
    """Did the member's last assistant message ask for a confirmation? Any message in between ends that."""
    last = session.scalar(
        select(Message.ended_by)
        .where(Message.family_id == ctx.family_id, Message.member_id == ctx.member_id, Message.role == "assistant")
        .order_by(Message.id.desc()).limit(1))
    return last == "confirmation"


def _failure(session: Session, action: PendingAction, exc: ServiceError) -> tuple[str, str]:
    if isinstance(exc, NotFoundError):
        return REPLY_NOT_FOUND, "not_found"
    if isinstance(exc, ForbiddenError):
        return REPLY_FORBIDDEN, "forbidden"
    if isinstance(exc, ConflictError):
        session.refresh(action)
        return (REPLY_EXPIRED if action.status == "expired" else REPLY_ALREADY_DONE), "conflict"
    return REPLY_FAILED, "error"


def answer_pending(session: Session, ctx: ToolContext, message: str, embedder: Any = None,
                   vector_store: Any = None) -> ConfirmationResult | None:
    """Run or cancel the member's pending action when `message` is a clear yes or no; otherwise None."""
    answer = classify(message)
    if answer is None:
        return None
    action = session.scalar(
        select(PendingAction)
        .where(PendingAction.family_id == ctx.family_id, PendingAction.member_id == ctx.member_id,
               PendingAction.status == "pending")
        .order_by(PendingAction.id.desc()).limit(1))
    if action is None:
        return None

    tool = "confirm_action" if answer == "yes" else "cancel_action"
    arguments = {"action_id": action.id, "tool": action.tool_name}
    started = time.perf_counter()
    try:
        if answer == "yes":
            result = execute_action(session, ctx, action.id, embedder, vector_store)
            verb = "عدّلت" if result["action"] == "edit_expense" else "حذفت"
            reply, ended_by = f"تم ✅ {verb}: {_line(result['expense'])}", "confirmed"
        else:
            result = cancel_action(session, ctx, action.id)
            reply, ended_by = REPLY_CANCELLED, "cancelled"
        status = "ok"
    except ServiceError as exc:
        reply, status = _failure(session, action, exc)
        ended_by, result = "confirm_error", exc.to_dict()
    latency_ms = int((time.perf_counter() - started) * 1000)
    return ConfirmationResult(reply, ended_by, tool, status, arguments, json.loads(dumps(result)), latency_ms)
