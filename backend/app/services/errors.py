"""Typed service errors. Their `code` becomes the tool result's error code (§7) and the API's (§8)."""
from collections.abc import Iterable


class ServiceError(Exception):
    code = "error"

    def __init__(self, message: str = "", fields: Iterable[str] | None = None) -> None:
        self.message = message or self.code
        self.fields = list(fields) if fields else None
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


class ForbiddenError(ServiceError):
    code = "forbidden"


class NotFoundError(ServiceError):
    code = "not_found"


class ValidationError(ServiceError):
    code = "invalid_args"


class MissingFieldsError(ServiceError):
    """Something the user has to provide (or a manager has to set) before the action can run."""

    code = "missing_fields"

    def __init__(self, fields: Iterable[str], message: str = "") -> None:
        fields = list(fields)
        super().__init__(message or f"missing: {', '.join(fields)}", fields)

    def to_dict(self) -> dict:
        return {**super().to_dict(), "missing_fields": self.fields}


class ConflictError(ServiceError):
    """The resource is not in a state that allows this (an invite already used, an action already executed)."""

    code = "conflict"


class AuthenticationError(ServiceError):
    code = "unauthorized"


class LockedError(ServiceError):
    """Too many failed logins for this username."""

    code = "locked"

    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after  # seconds


class UnavailableError(ServiceError):
    """The server cannot do this now because of its own configuration (a missing key or secret)."""

    code = "not_available"
