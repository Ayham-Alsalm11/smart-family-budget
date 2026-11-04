"""Registration, login and the current member (§5, §8)."""
from decimal import Decimal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from app.auth.dependencies import AppSettings, Context, DbSession, jwt_configured
from app.auth.tokens import create_token
from app.db.models import Member
from app.services import accounts
from app.settings import Settings

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(jwt_configured)])
me_router = APIRouter(tags=["auth"])


class Account(BaseModel):
    name: str
    username: str
    password: str


class RegisterFamily(Account):
    family_name: str
    monthly_income: Decimal
    base_currency: str = "SYP_NEW"
    usd_rate: Decimal | None = None


class Join(Account):
    code: str


class Login(BaseModel):
    username: str
    password: str


def _session_reply(settings: Settings, member: Member) -> dict:
    return {"access_token": create_token(settings, member.id), "token_type": "bearer",
            "member": accounts.member_to_dict(member)}


@router.post("/register-family", status_code=status.HTTP_201_CREATED)
def register_family(body: RegisterFamily, session: DbSession, settings: AppSettings):
    return _session_reply(settings, accounts.register_family(session, **body.model_dump()))


@router.post("/join", status_code=status.HTTP_201_CREATED)
def join(body: Join, session: DbSession, settings: AppSettings):
    return _session_reply(settings, accounts.join_with_code(session, **body.model_dump()))


@router.post("/login")
def login(body: Login, session: DbSession, settings: AppSettings):
    member = accounts.login(session, body.username, body.password, max_failures=settings.login_max_failures,
                            lock_minutes=settings.login_lock_minutes)
    return _session_reply(settings, member)


@me_router.get("/me")
def me(session: DbSession, ctx: Context):
    return accounts.profile(session, ctx)
