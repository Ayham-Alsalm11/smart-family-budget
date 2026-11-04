"""FastAPI dependencies: the database session, the current member and the ToolContext.

The context comes only from the token and the database (D7). No request body or query can set the family,
the member or the role.
"""
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.auth.tokens import INVALID_TOKEN, read_token, require_secret
from app.db.models import Member
from app.services.errors import AuthenticationError
from app.settings import Settings

_bearer = HTTPBearer(auto_error=False)


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


AppSettings = Annotated[Settings, Depends(get_app_settings)]
DbSession = Annotated[Session, Depends(get_session)]


def jwt_configured(settings: AppSettings) -> None:
    """Login paths refuse to work without a usable JWT_SECRET, before they change anything."""
    require_secret(settings)


def current_member(
    settings: AppSettings,
    session: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Member:
    if credentials is None:
        raise AuthenticationError("سجّل الدخول أولاً.")
    member = session.get(Member, read_token(settings, credentials.credentials))
    if member is None or not member.is_active:
        raise AuthenticationError(INVALID_TOKEN)
    return member


CurrentMember = Annotated[Member, Depends(current_member)]


def tool_context(settings: AppSettings, member: CurrentMember) -> ToolContext:
    return ToolContext(family_id=member.family_id, member_id=member.id, role=member.role, today=settings.today(),
                       model=settings.model_id(settings.agent_model))


Context = Annotated[ToolContext, Depends(tool_context)]
