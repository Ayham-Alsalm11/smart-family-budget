"""Conversation memory (§6.8) and the link between a clarification and its answer (§6.6).

The model sees the member's own recent messages, in time order, between the system prompt and the new message.
Assistant messages carry their `tool_summary` in brackets, so that "امسح التاني" can be resolved to an expense id.
The ids that reach the model this way are also what `known_expense_ids` accepts for an edit or a deletion (§6.5).
"""
import re
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.db.models import Message, utcnow
from app.llm.client import estimate_prompt_tokens

MAX_REPLY_CHARS = 300
LINK = " ← "
_SUMMARY_ID = re.compile(r"#(\d+)")
# Only a marked number is an id: a bare one in "امسح مصروف 5000" is an amount.
_NAMED_ID = re.compile(r"(?:#|رقم|\bid\b)\s*:?\s*(\d+)", re.IGNORECASE)


def recent_messages(session: Session, ctx: ToolContext, count: int, hours: int) -> list[Message]:
    """The member's last `count` messages of the last `hours` hours, oldest first. No other member's, ever."""
    if count <= 0:
        return []
    rows = session.scalars(
        select(Message)
        .where(Message.family_id == ctx.family_id, Message.member_id == ctx.member_id,
               Message.created_at >= utcnow() - timedelta(hours=hours))
        .order_by(Message.id.desc()).limit(count)
    ).all()
    return list(reversed(rows))


def _to_chat(row: Message) -> dict:
    if row.role == "user":
        return {"role": "user", "content": row.content}
    content = row.content
    if len(content) > MAX_REPLY_CHARS:
        content = content[:MAX_REPLY_CHARS] + "…"
    if row.tool_summary:
        content = f"{content} ({row.tool_summary})"
    return {"role": "assistant", "content": content}


def memory_for_model(history: list[Message], max_tokens: int) -> list[dict]:
    """`history` as chat messages, without the oldest ones when they pass the rough token ceiling."""
    chat = [_to_chat(row) for row in history]
    while chat and estimate_prompt_tokens(chat) > max_tokens:
        chat.pop(0)
    return chat


def known_expense_ids(history: list[Message], memory: list[dict], message: str) -> set[int]:
    """The ids the conversation has shown so far: in the summaries of the messages actually sent, and in the user's
    own words. `memory` may be shorter than `history`, because the token ceiling drops the oldest messages."""
    sent = history[-len(memory):] if memory else []
    ids = {int(found) for row in sent for found in _SUMMARY_ID.findall(row.tool_summary or "")}
    return ids | {int(found) for found in _NAMED_ID.findall(message)}


def linked_source_text(history: list[Message], message: str) -> str:
    """After a clarification: "<the original message> ← <the answer>", kept for the correction memory (§6.7).

    A second question in a row adds to the chain: "دفعت 50 ← لأبو أحمد ← كانت حق خضرا".
    """
    parts = [message]
    index = len(history) - 1
    while index >= 1 and history[index].ended_by == "clarification" and history[index - 1].role == "user":
        parts.insert(0, history[index - 1].content)
        index -= 2
    return LINK.join(parts)
