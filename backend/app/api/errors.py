"""One error shape for the whole API (§8): {"error": {"code", "message", "fields"?}}, with the tools' codes."""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.services.errors import (
    AuthenticationError, ConflictError, ForbiddenError, LockedError, MissingFieldsError, NotFoundError, ServiceError,
    UnavailableError, ValidationError,
)

log = logging.getLogger(__name__)

STATUS = {
    AuthenticationError: 401,
    ForbiddenError: 403,
    NotFoundError: 404,
    ConflictError: 409,
    ValidationError: 422,
    MissingFieldsError: 422,
    LockedError: 429,
    UnavailableError: 503,
}
HTTP_CODES = {401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed"}
_LOCATIONS = ("body", "query", "path", "header")


def error_response(status: int, code: str, message: str, fields: list[str] | None = None,
                   headers: dict[str, str] | None = None) -> JSONResponse:
    error: dict = {"code": code, "message": message}
    if fields:
        error["fields"] = fields
    return JSONResponse({"error": error}, status_code=status, headers=headers)


def _service_error(_request: Request, exc: ServiceError) -> JSONResponse:
    headers = None
    if isinstance(exc, LockedError):
        headers = {"Retry-After": str(exc.retry_after)}
    elif isinstance(exc, AuthenticationError):
        headers = {"WWW-Authenticate": "Bearer"}
    return error_response(STATUS.get(type(exc), 500), exc.code, exc.message, exc.fields, headers)


def _request_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Field names and reasons only: the rejected values are never echoed, because they can be passwords."""
    errors = exc.errors()
    fields, reasons = [], []
    for error in errors:
        location = [str(part) for part in error["loc"] if part not in _LOCATIONS]
        field = ".".join(location)
        fields.append(field)
        reasons.append(f"{field}: {error['msg']}" if field else error["msg"])
    missing_only = all(error["type"] == "missing" for error in errors)
    return error_response(422, "missing_fields" if missing_only else "invalid_args", "; ".join(reasons),
                          [f for f in fields if f])


def _http_error(_request: Request, exc: HTTPException) -> JSONResponse:
    return error_response(exc.status_code, HTTP_CODES.get(exc.status_code, "error"), str(exc.detail),
                          headers=exc.headers)


def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    log.error("%s %s failed", request.method, request.url.path, exc_info=exc)
    return error_response(500, "error", "internal error")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ServiceError, _service_error)
    app.add_exception_handler(RequestValidationError, _request_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected_error)
