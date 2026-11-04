"""Pending edits and deletions (§6.5). Step 5a creates them; confirming them comes in step 6."""
from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.db.models import PendingAction, utcnow
from app.services.currency import parse_amount, validate_currency
from app.services.errors import ValidationError
from app.services.expenses import EDITABLE_FIELDS, get_category, get_expense
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
