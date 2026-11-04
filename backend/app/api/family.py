"""Family management, for managers (§8). The checks are in `services/accounts.py`."""
from decimal import Decimal

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.api.deps import ResourceId
from app.auth.dependencies import AppSettings, Context, DbSession
from app.services import accounts

router = APIRouter(tags=["family"])


class NewInvite(BaseModel):
    role: str = "member"


class NewRole(BaseModel):
    role: str


class FamilyChanges(BaseModel):
    name: str | None = None
    monthly_income: Decimal | None = None
    base_currency: str | None = None
    usd_rate: Decimal | None = None


@router.get("/members")
def list_members(session: DbSession, ctx: Context):
    return accounts.list_members(session, ctx)


@router.delete("/members/{member_id}")
def deactivate_member(member_id: ResourceId, session: DbSession, ctx: Context):
    return accounts.deactivate_member(session, ctx, member_id)


@router.put("/members/{member_id}/role")
def change_role(member_id: ResourceId, body: NewRole, session: DbSession, ctx: Context):
    return accounts.change_role(session, ctx, member_id, body.role)


@router.post("/invites", status_code=status.HTTP_201_CREATED)
def create_invite(session: DbSession, ctx: Context, settings: AppSettings, body: NewInvite | None = None):
    return accounts.create_invite(session, ctx, (body or NewInvite()).role, settings.invite_days)


@router.get("/invites")
def list_invites(session: DbSession, ctx: Context):
    return accounts.list_invites(session, ctx)


@router.delete("/invites/{invite_id}")
def revoke_invite(invite_id: ResourceId, session: DbSession, ctx: Context):
    return accounts.revoke_invite(session, ctx, invite_id)


@router.put("/family")
def update_family(body: FamilyChanges, session: DbSession, ctx: Context):
    return accounts.update_family(session, ctx, body.model_dump(exclude_unset=True))
