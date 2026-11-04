"""Confirm and cancel buttons for pending edits and deletions (§6.5)."""
from fastapi import APIRouter

from app.api.deps import ResourceId, SearchServices
from app.auth.dependencies import Context, DbSession
from app.services import pending

router = APIRouter(prefix="/actions", tags=["actions"])


@router.post("/{action_id}/confirm")
def confirm(action_id: ResourceId, session: DbSession, ctx: Context, rag: SearchServices):
    return pending.execute_action(session, ctx, action_id, rag.embedder, rag.vector_store)


@router.post("/{action_id}/cancel")
def cancel(action_id: ResourceId, session: DbSession, ctx: Context):
    return pending.cancel_action(session, ctx, action_id)
