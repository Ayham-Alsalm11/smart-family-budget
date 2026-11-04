"""Password rules and bcrypt hashing (§5). The rules apply when a password is chosen, not at login."""
from functools import lru_cache

import bcrypt

from app.services.errors import ValidationError

MIN_CHARACTERS = 8
MAX_BYTES = 72  # bcrypt's limit, in UTF-8: an Arabic letter takes 2 bytes
ROUNDS = 12


def _encoded(password: str) -> bytes | None:
    """UTF-8 bytes, or None for text that has no UTF-8 form (a lone surrogate from JSON)."""
    try:
        return password.encode("utf-8")
    except UnicodeEncodeError:
        return None


def validate_password(password: str) -> None:
    if len(password) < MIN_CHARACTERS:
        raise ValidationError(f"كلمة المرور قصيرة: {MIN_CHARACTERS} أحرف على الأقل.", ["password"])
    encoded = _encoded(password)
    if encoded is None:
        raise ValidationError("كلمة المرور فيها رموز غير صالحة.", ["password"])
    if len(encoded) > MAX_BYTES:
        raise ValidationError(f"كلمة المرور طويلة: الحد {MAX_BYTES} بايت، والحرف العربي بايتان.", ["password"])


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(ROUNDS)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    encoded = _encoded(password)
    if encoded is None or len(encoded) > MAX_BYTES:  # bcrypt raises on these; no valid password looks like them
        return False
    return bcrypt.checkpw(encoded, password_hash.encode("ascii"))


@lru_cache
def _unusable_hash(rounds: int) -> str:
    return bcrypt.hashpw(b"no account", bcrypt.gensalt(rounds)).decode("ascii")


def burn_verification_time(password: str) -> None:
    """Spend what a real check costs, so that an unknown username does not answer faster."""
    verify_password(password, _unusable_hash(ROUNDS))
