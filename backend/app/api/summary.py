"""The dashboard (§8). Savings goals join in step 8."""
from fastapi import APIRouter

from app.auth.dependencies import Context, DbSession
from app.services import reports

router = APIRouter(tags=["summary"])


@router.get("/summary")
def summary(session: DbSession, ctx: Context, month: str | None = None):
    return reports.monthly_report(session, ctx, month)
