"""Typed service errors. Their `code` becomes the tool result's error code (§7)."""
from collections.abc import Iterable


class ServiceError(Exception):
    code = "error"

    def __init__(self, message: str = "") -> None:
        self.message = message or self.code
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
        self.fields = list(fields)
        super().__init__(message or f"missing: {', '.join(self.fields)}")

    def to_dict(self) -> dict:
        return {**super().to_dict(), "missing_fields": self.fields}
