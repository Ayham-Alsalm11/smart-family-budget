"""Pending edits and deletions (§6.5): created by the tools, executed or cancelled here.

`execute_action` is the only way a pending action runs. The confirm button uses it now; confirmation in chat
will use the same function in step 6.
"""
from datetime import timedelta
from typing import Any

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.db.models import PendingAction, utcnow
from app.services.currency import parse_amount, validate_currency
from app.services.errors import ConflictError, ForbiddenError, NotFoundError, ServiceError, ValidationError
from app.services.expenses import (
    EDITABLE_FIELDS, Embedder, VectorStore, delete_expense, get_category, get_expense, update_expense,
)
from app.services.periods import parse_date

CONFIRMATION_MINUTES = 10


def _validate_changes(session: Session, changes: dict[str, Any]) -> None:
    """The same checks update_expense will apply, so that a pending edit is valid when confirmed."""
    if not changes:
        raise ValidationError("no changes given")
    unknown = sorted(set(changes) - EDITABLE_FIELDS)
    if unknown:
        raise ValidationError(f"these fields cannot be changed: {', '.join(unknown)}")
    if "amount" in changes:
        parse_amount(changes["amount"])
    if "category" in changes:
        get_category(session, changes["category"])
    if "currency" in changes:
        validate_currency(changes["currency"])
    if "date" in changes:
        parse_date(changes["date"])


def request_change(
    session: Session,
    ctx: ToolContext,
    tool_name: str,
    expense_id: int,
    changes: dict[str, Any] | None = None,
) -> dict:
    """Check ownership (§5), store the action as pending, and return a preview with its confirmation_id."""
    current = get_expense(session, ctx, expense_id)  # raises NotFoundError / ForbiddenError
    arguments: dict[str, Any] = {"expense_id": expense_id}
    if tool_name == "edit_expense":
        _validate_changes(session, changes or {})
        arguments["changes"] = changes
    elif tool_name != "delete_expense":
        raise ValueError(f"no confirmation flow for {tool_name!r}")

    action = PendingAction(
        family_id=ctx.family_id, member_id=ctx.member_id, message_id=ctx.message_id, tool_name=tool_name,
        arguments=arguments, status="pending", expires_at=utcnow() + timedelta(minutes=CONFIRMATION_MINUTES),
    )
    session.add(action)
    session.commit()
    preview = {key: current[key] for key in ("id", "amount", "currency", "category", "expense_date", "description")}
    return {"confirmation_id": action.id, "action": tool_name, "expense": preview, "changes": changes}


def _open_action(session: Session, ctx: ToolContext, action_id: int) -> PendingAction:
    """The caller's own action, still pending and not past its time. A late one becomes `expired` for good."""
    action = session.get(PendingAction, action_id)
    if action is None or action.family_id != ctx.family_id:
        raise NotFoundError(f"action #{action_id} not found")
    if action.member_id != ctx.member_id:
        raise ForbiddenError("only the member who asked for this action can confirm or cancel it")
    if action.status == "pending" and action.expires_at <= utcnow():
        action.status = "expired"
        session.commit()
    if action.status != "pending":
        raise ConflictError(f"action #{action_id} is already {action.status}")
    return action


def _claim(session: Session, action: PendingAction, status: str) -> None:
    """One conditional UPDATE, so that two parallel confirmations cannot both run."""
    claimed = session.execute(
        update(PendingAction).where(PendingAction.id == action.id, PendingAction.status == "pending")
        .values(status=status)
    ).rowcount
    if not claimed:
        session.rollback()
        raise ConflictError(f"action #{action.id} is not pending any more")


def execute_action(
    session: Session,
    ctx: ToolContext,
    action_id: int,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
) -> dict:
    """Run a pending edit or deletion. `update_expense` and `delete_expense` check again that the permission
    still holds; if they refuse, nothing changes and the action stays pending until it expires."""
    action = _open_action(session, ctx, action_id)
    _claim(session, action, "confirmed")
    arguments = action.arguments
    try:
        if action.tool_name == "edit_expense":
            result = update_expense(session, ctx, arguments["expense_id"], arguments["changes"], embedder, vector_store)
        else:
            result = delete_expense(session, ctx, arguments["expense_id"], vector_store)
    except ServiceError:
        session.rollback()
        raise
    return {"id": action_id, "status": "confirmed", "action": action.tool_name, "expense": result}


def cancel_action(session: Session, ctx: ToolContext, action_id: int) -> dict:
    action = _open_action(session, ctx, action_id)
    _claim(session, action, "cancelled")
    session.commit()
    return {"id": action_id, "status": "cancelled", "action": action.tool_name}
