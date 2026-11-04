"""The request context passed to every service and tool (§6.2)."""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ToolContext:
    """Who is asking, and on which day. Built by the server from the login token; never chosen by the LLM."""

    family_id: int
    member_id: int
    role: str
    today: date
    model: str | None = None
    message_id: int | None = None

    @property
    def is_manager(self) -> bool:
        return self.role == "manager"
