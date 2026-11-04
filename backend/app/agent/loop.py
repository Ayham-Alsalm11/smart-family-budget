"""The ReAct loop (§6.1, §6.2): plain Python with native function calling, traceable step by step.

One turn = at most `max_steps` LLM steps. Tool calls of one step run in order; invalid ones are not
executed and their error goes back to the model; `ask_clarification` ends the turn. Every LLM attempt
goes to `llm_calls`, every tool call to `tool_log`, and the user message and the reply to `messages`.
Sending earlier turns to the model (§6.8) comes in step 6.
"""
import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.orm import Session

from app.agent.context import ToolContext
from app.agent.grounding import ungrounded_numbers
from app.agent.prompts import build_system_prompt
from app.db.models import Family, LlmCall, Message, ToolLog
from app.llm.client import Attempt, LLMError, LLMToolValidationError, Usage
from app.tools.dispatch import ToolDeps, dispatch
from app.tools.schemas import TOOL_SCHEMAS

REPLY_STEP_LIMIT = "عذراً، ما قدرت كمّل هالطلب. فيك تكتبه بشكل أبسط؟"
REPLY_LLM_ERROR = "عذراً، في مشكلة بالاتصال هلق. جرّب كمان شوي."
REPLY_EMPTY = "عذراً، ما فهمت عليك. فيك توضّح أكتر؟"
REPLY_TOOL_REJECTED = "عذراً، ما قدرت نفّذ طلبك بالشكل الصحيح. فيك تكتبه بطريقة أوضح؟"
MAX_TOOL_CORRECTIONS = 2  # provider-side rejections (tool_use_failed) the model may fix in one turn
_REJECTED_TOOL = re.compile(r"for tool (\w+)")
_REJECTION_PREFIX = re.compile(r"^(?:tool_use_failed:\s*)?(?:tool call validation failed:\s*)+", re.IGNORECASE)


def _rejection(error: LLMToolValidationError) -> tuple[str, str]:
    """(tool name or 'unknown', short reason) from Groq's tool_use_failed message."""
    text = str(error)
    match = _REJECTED_TOOL.search(text)
    return (match.group(1) if match else "unknown"), _REJECTION_PREFIX.sub("", text).strip()[:300]


@dataclass
class StepRecord:
    step: int
    tool: str
    arguments: Any
    status: str
    latency_ms: int
    result: dict

    def as_dict(self) -> dict:
        return {"step": self.step, "tool": self.tool, "arguments": self.arguments, "status": self.status,
                "latency_ms": self.latency_ms}


@dataclass
class TurnResult:
    reply: str
    ended_by: str  # reply, clarification, step_limit, llm_error, tool_rejected
    steps: list[StepRecord] = field(default_factory=list)
    pending_action: dict | None = None
    alerts: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    llm_calls: list[dict] = field(default_factory=list)
    llm_steps: int = 0
    ungrounded: list[str] = field(default_factory=list)
    message_id: int | None = None
    error: str | None = None


def _summary(steps: list[StepRecord]) -> str:
    parts = []
    for s in steps:
        if s.tool == "add_expense" and s.status == "ok" and "data" in s.result:
            parts.append(f"add_expense → #{s.result['data']['expense']['id']}")
        else:
            parts.append(f"{s.tool} → {s.status}")
    return "; ".join(parts)


def run_turn(
    session: Session,
    ctx: ToolContext,
    message: str,
    client: Any,
    *,
    model: str | None = None,
    embedder: Any = None,
    vector_store: Any = None,
    max_steps: int = 6,
    corrections: list[str] | None = None,
) -> TurnResult:
    model = model or ctx.model
    family = session.get(Family, ctx.family_id)
    user_row = Message(family_id=ctx.family_id, member_id=ctx.member_id, role="user", content=message)
    session.add(user_row)
    session.commit()
    ctx = replace(ctx, message_id=user_row.id, model=model)
    deps = ToolDeps(session, source_text=message, embedder=embedder, vector_store=vector_store)

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(ctx.today, family.base_currency, corrections)},
        {"role": "user", "content": message},
    ]
    turn = TurnResult(reply="", ended_by="", message_id=user_row.id)
    tool_texts: list[str] = []
    corrections = 0

    for step in range(1, max_steps + 1):
        turn.llm_steps = step

        def record(attempt: Attempt, step: int = step) -> None:
            usage = attempt.usage or Usage()
            session.add(LlmCall(
                message_id=ctx.message_id, model=attempt.model, step=step, attempt=attempt.attempt,
                status=attempt.status, http_status=attempt.http_status, latency_ms=attempt.latency_ms,
                prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                reasoning_tokens=usage.reasoning_tokens, cached_tokens=usage.cached_tokens, error=attempt.error,
            ))
            turn.llm_calls.append({"step": step, "attempt": attempt.attempt, "status": attempt.status,
                                   "http_status": attempt.http_status, "latency_ms": attempt.latency_ms,
                                   "waited_ms": attempt.waited_ms, "tokens": usage.as_dict()})

        try:
            response = client.chat(model, messages, TOOL_SCHEMAS, on_attempt=record)
        except LLMToolValidationError as exc:
            # Not a connection failure: the model's call broke the schema. It goes back to the model (§6.2).
            tool, reason = _rejection(exc)
            result = {"ok": False, "error": {"code": "invalid_args", "message": reason}}
            session.add(ToolLog(
                message_id=ctx.message_id, family_id=ctx.family_id, member_id=ctx.member_id, step=step,
                tool_name=tool[:50], arguments={"rejected_by_provider": reason}, result=result,
                status="invalid_args", model=model,
            ))
            turn.steps.append(StepRecord(step, tool, {"rejected_by_provider": reason}, "invalid_args", 0, result))
            if corrections >= MAX_TOOL_CORRECTIONS:
                turn.reply, turn.ended_by, turn.error = REPLY_TOOL_REJECTED, "tool_rejected", f"{exc.kind}: {reason}"
                break
            corrections += 1
            messages.append({"role": "user", "content": f"Your last tool call was rejected: {reason}. "
                                                        "Fix the arguments or omit optional ones."})
            continue
        except LLMError as exc:
            turn.reply, turn.ended_by, turn.error = REPLY_LLM_ERROR, "llm_error", f"{exc.kind}: {exc}"
            break
        finally:
            session.commit()
        turn.usage += response.usage

        if not response.tool_calls:
            turn.reply = (response.content or "").strip() or REPLY_EMPTY
            turn.ended_by = "reply"
            break

        messages.append(response.assistant_message)
        for call in response.tool_calls:
            started = time.perf_counter()
            result = dispatch(call.name, call.arguments, ctx, deps, call.error)
            latency_ms = int((time.perf_counter() - started) * 1000)
            text = result.to_json()
            stored = json.loads(text)
            arguments = call.arguments if call.arguments is not None else {"raw": call.raw_arguments}
            session.add(ToolLog(
                message_id=ctx.message_id, family_id=ctx.family_id, member_id=ctx.member_id, step=step,
                tool_name=call.name[:50], arguments=arguments, result=stored, status=result.log_status,
                latency_ms=latency_ms, model=model,
            ))
            turn.steps.append(StepRecord(step, call.name, arguments, result.code, latency_ms, stored))
            tool_texts.append(text)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

            if result.ok and call.name == "add_expense":
                turn.alerts.extend(stored["data"]["alerts"])
            if result.code == "confirmation_required":
                turn.pending_action = {k: stored["error"].get(k) for k in ("confirmation_id", "action", "expense",
                                                                           "changes")}
            if result.ok and call.name == "ask_clarification":
                turn.reply, turn.ended_by = result.data["question"], "clarification"
                break
        session.commit()
        if turn.ended_by:
            break
    else:
        turn.reply, turn.ended_by = REPLY_STEP_LIMIT, "step_limit"

    if turn.ended_by == "reply":
        turn.ungrounded = [str(n) for n in ungrounded_numbers(turn.reply, tool_texts)]
        if turn.ungrounded:
            session.add(ToolLog(
                message_id=ctx.message_id, family_id=ctx.family_id, member_id=ctx.member_id, step=turn.llm_steps,
                tool_name="grounding", arguments={"numbers": turn.ungrounded}, result=None, status="ungrounded",
                model=model,
            ))
    session.add(Message(family_id=ctx.family_id, member_id=ctx.member_id, role="assistant", content=turn.reply,
                        tool_summary=_summary(turn.steps) or None))
    session.commit()
    return turn
