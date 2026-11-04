"""POST /chat and GET /messages (§8). The model is reached through the one client in app.state."""
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select

from app.agent.loop import run_turn
from app.api.deps import SearchServices
from app.auth.dependencies import AppSettings, Context, DbSession
from app.db.models import Message
from app.services.errors import UnavailableError

router = APIRouter(tags=["chat"])

MAX_MESSAGE = 1000
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE)]


class Chat(BaseModel):
    message: MessageText


@router.post("/chat")
def chat(body: Chat, request: Request, session: DbSession, ctx: Context, settings: AppSettings,
         rag: SearchServices):
    client = request.app.state.llm
    if client is None:
        raise UnavailableError("المحادثة متوقفة: مفتاح Groq غير موجود في إعدادات الخادم.")
    turn = run_turn(session, ctx, body.message, client, embedder=rag.embedder, vector_store=rag.vector_store,
                    max_steps=settings.llm_max_steps)
    reply = {"reply": turn.reply, "steps": [step.as_dict() for step in turn.steps], "alerts": turn.alerts,
             "message_id": turn.message_id}
    if turn.pending_action:
        reply["pending_action"] = turn.pending_action
    return reply


@router.get("/messages")
def messages(session: DbSession, ctx: Context, limit: Annotated[int, Query(ge=1, le=200)] = 50):
    """The caller's own conversation: the last `limit` messages, oldest first."""
    rows = session.scalars(
        select(Message).where(Message.family_id == ctx.family_id, Message.member_id == ctx.member_id)
        .order_by(Message.id.desc()).limit(limit)
    ).all()
    return {"messages": [{"id": m.id, "role": m.role, "content": m.content, "tool_summary": m.tool_summary,
                          "created_at": m.created_at.isoformat()} for m in reversed(rows)]}
