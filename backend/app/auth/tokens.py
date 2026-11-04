"""JWT (§5): HS256 with the member's id in `sub`, plus `iat` and `exp`. Nothing else is trusted from the token:
the family and the role are read from the database on every request."""
from datetime import datetime, timedelta

import jwt

from app.db.models import utcnow
from app.services.errors import AuthenticationError, UnavailableError
from app.settings import Settings

ALGORITHM = "HS256"
MIN_SECRET_LENGTH = 32
INVALID_TOKEN = "جلسة الدخول غير صالحة أو منتهية. سجّل الدخول من جديد."


def require_secret(settings: Settings) -> str:
    secret = settings.jwt_secret.get_secret_value() if settings.jwt_secret else ""
    if len(secret) < MIN_SECRET_LENGTH:
        raise UnavailableError(f"الدخول متوقف: JWT_SECRET في إعدادات الخادم فارغ أو أقصر من {MIN_SECRET_LENGTH} حرفاً.")
    return secret


def create_token(settings: Settings, member_id: int, now: datetime | None = None) -> str:
    now = now or utcnow()
    claims = {"sub": str(member_id), "iat": now, "exp": now + timedelta(days=settings.jwt_expire_days)}
    return jwt.encode(claims, require_secret(settings), algorithm=ALGORITHM)


def read_token(settings: Settings, token: str) -> int:
    """The member id of a valid token. Anything else (expired, altered, malformed) raises AuthenticationError."""
    try:
        claims = jwt.decode(token, require_secret(settings), algorithms=[ALGORITHM],
                            options={"require": ["sub", "iat", "exp"]})
        return int(claims["sub"])
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise AuthenticationError(INVALID_TOKEN) from exc
