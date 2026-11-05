"""The ReAct loop (§6.1, §6.2): plain Python with native function calling, traceable step by step.

One turn = at most `max_steps` LLM steps. Tool calls of one step run in order; invalid ones are not
executed and their error goes back to the model; `ask_clarification` and `confirmation_required` end the
turn. Every LLM attempt goes to `llm_calls`, every tool call to `tool_log`, and the user message and the reply
to `messages`. Earlier turns reach the model through `agent/memory.py` (§6.8). A yes or no to a pending action
never reaches the model at all (§6.5, `agent/confirm.py`).
"""
import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.orm import Session

from app.agent.confirm import ConfirmationResult, answer_pending, preview_text, waiting_for_an_answer
from app.agent.context import ToolContext
from app.agent.grounding import ungrounded_numbers
from app.agent.memory import known_expense_ids, linked_source_text, memory_for_model, recent_messages
from app.agent.prompts import build_system_prompt
from app.db.models import Family, LlmCall, Message, ToolLog
from app.llm.client import Attempt, LLMError, LLMToolValidationError, LLMTruncatedError, Usage
from app.tools.dispatch import ToolDeps, dispatch
from app.tools.schemas import TOOL_SCHEMAS

REPLY_STEP_LIMIT = "عذراً، ما قدرت كمّل هالطلب. فيك تكتبه بشكل أبسط؟"
REPLY_LLM_ERROR = "عذراً، في مشكلة بالاتصال هلق. جرّب كمان شوي."
REPLY_EMPTY = "عذراً، ما فهمت عليك. فيك توضّح أكتر؟"
REPLY_TRUNCATED = "عذراً، الرد طلع طويل وانقطع. فيك تطلبه بشكل أقصر؟"
REPLY_TOOL_REJECTED = "عذراً، ما قدرت نفّذ طلبك بالشكل الصحيح. فيك تكتبه بطريقة أوضح؟"
MAX_TOOL_CORRECTIONS = 2  # provider-side rejections (tool_use_failed) the model may fix in one turn
_REJECTED_TOOL = re.compile(r"for tool (\w+)")
_REJECTION_PREFIX = re.compile(r"^(?:tool_use_failed:\s*)?(?:tool call validation failed:\s*)+", re.IGNORECASE)
SUMMARY_MAX_IDS = 5
REPAIR_REQUEST = ("Your reply has numbers that are not in the tool results: {numbers}. Rewrite it using only numbers "
                  "from the tool results, or call a tool to get them. If a number is not available, leave it out.")


@dataclass(frozen=True)
class TurnOptions:
    memory_messages: int = 10
    memory_hours: int = 24
    memory_max_tokens: int = 600
    grounding_repair: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> "TurnOptions":
        return cls(settings.memory_messages, settings.memory_hours, settings.memory_max_tokens,
                   settings.grounding_repair)


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
    ended_by: str  # one of MESSAGE_OUTCOMES in db/models.py
    steps: list[StepRecord] = field(default_factory=list)
    pending_action: dict | None = None
    alerts: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    llm_calls: list[dict] = field(default_factory=list)
    llm_steps: int = 0
    ungrounded: list[str] = field(default_factory=list)  # in the final reply
    repaired: bool = False  # a repair was attempted (§6.9)
    first_reply: str | None = None  # the reply before the repair
    first_ungrounded: list[str] = field(default_factory=list)
    message_id: int | None = None
    error: str | None = None


def _ungrounded(reply: str, tool_texts: list[str], message: str) -> list[str]:
    return [str(number) for number in ungrounded_numbers(reply, tool_texts, message)]


def _expense_ids(step: StepRecord) -> list[int]:
    if step.status != "ok":
        return []
    data = step.result.get("data") or {}
    if step.tool == "add_expense":
        return [data["expense"]["id"]]
    if step.tool == "list_expenses":
        return [row["id"] for row in data["expenses"]]
    if step.tool == "search_expenses":
        return [row["id"] for row in data["matches"]]
    return []


def _summary(steps: list[StepRecord]) -> str:
    """Goes to the model with the reply (§6.8): the ids are what "امسح التاني" refers to."""
    parts = []
    for s in steps:
        ids = _expense_ids(s)[:SUMMARY_MAX_IDS]
        parts.append(f"{s.tool} → {', '.join(f'#{i}' for i in ids) if ids else s.status}")
    return "; ".join(parts)


def _finish_without_model(session: Session, ctx: ToolContext, done: ConfirmationResult) -> TurnResult:
    step = StepRecord(0, done.tool, done.arguments, done.status, done.latency_ms, done.result)
    session.add(ToolLog(
        message_id=ctx.message_id, family_id=ctx.family_id, member_id=ctx.member_id, step=0, tool_name=done.tool,
        arguments=done.arguments, result=done.result, status=done.status, latency_ms=done.latency_ms,
    ))
    session.add(Message(family_id=ctx.family_id, member_id=ctx.member_id, role="assistant", content=done.reply,
                        tool_summary=_summary([step]), ended_by=done.ended_by))
    session.commit()
    return TurnResult(reply=done.reply, ended_by=done.ended_by, steps=[step], message_id=ctx.message_id)


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
    options: TurnOptions = TurnOptions(),
) -> TurnResult:
    model = model or ctx.model
    family = session.get(Family, ctx.family_id)
    history = recent_messages(session, ctx, options.memory_messages, options.memory_hours)
    waiting = waiting_for_an_answer(session, ctx)
    user_row = Message(family_id=ctx.family_id, member_id=ctx.member_id, role="user", content=message)
    session.add(user_row)
    session.commit()
    ctx = replace(ctx, message_id=user_row.id, model=model)
    if waiting:
        done = answer_pending(session, ctx, message, embedder, vector_store)
        if done is not None:
            return _finish_without_model(session, ctx, done)

    memory = memory_for_model(history, options.memory_max_tokens)
    deps = ToolDeps(session, source_text=linked_source_text(history, message), embedder=embedder,
                    vector_store=vector_store, known_expense_ids=known_expense_ids(history, memory, message))

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(ctx.today, family.base_currency, corrections)},
        *memory,
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
                                   "waited_ms": attempt.waited_ms, "tokens": usage.as_dict(),
                                   "retry_after": attempt.retry_after, "error": attempt.error})

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
        except LLMTruncatedError as exc:
            turn.usage += exc.usage or Usage()
            turn.reply, turn.ended_by, turn.error = REPLY_TRUNCATED, "llm_error", f"{exc.kind}: {exc}"
            break
        except LLMError as exc:
            turn.reply, turn.ended_by, turn.error = REPLY_LLM_ERROR, "llm_error", f"{exc.kind}: {exc}"
            break
        finally:
            session.commit()
        turn.usage += response.usage

        if not response.tool_calls:
            turn.reply = (response.content or "").strip() or REPLY_EMPTY
            flagged = _ungrounded(turn.reply, tool_texts, message)
            if flagged and options.grounding_repair and not turn.repaired and step < max_steps:
                turn.repaired, turn.first_reply, turn.first_ungrounded = True, turn.reply, flagged
                messages.append(response.assistant_message)
                messages.append({"role": "user", "content": REPAIR_REQUEST.format(numbers=", ".join(flagged))})
                continue
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
            deps.known_expense_ids.update(_expense_ids(turn.steps[-1]))
            tool_texts.append(text)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

            if result.ok and call.name == "add_expense":
                turn.alerts.extend(stored["data"]["alerts"])
            if result.code == "confirmation_required":
                turn.pending_action = {k: stored["error"].get(k) for k in ("confirmation_id", "action", "expense",
                                                                           "changes")}
                turn.reply, turn.ended_by = preview_text(turn.pending_action), "confirmation"
                break
            if result.ok and call.name == "ask_clarification":
                turn.reply, turn.ended_by = result.data["question"], "clarification"
                break
        session.commit()
        if turn.ended_by:
            break
    else:
        turn.reply, turn.ended_by = REPLY_STEP_LIMIT, "step_limit"

    if turn.repaired and turn.ended_by in ("step_limit", "llm_error", "tool_rejected"):
        # The turn's work is done and only the repair failed: the first reply is better than an apology.
        turn.reply, turn.ended_by, turn.error = turn.first_reply, "reply", None
    if turn.ended_by == "reply":
        turn.ungrounded = _ungrounded(turn.reply, tool_texts, message)
        if turn.repaired:
            session.add(ToolLog(
                message_id=ctx.message_id, family_id=ctx.family_id, member_id=ctx.member_id, step=turn.llm_steps,
                tool_name="grounding_repair", result=None, status="ungrounded" if turn.ungrounded else "ok", model=model,
                arguments={"first_reply": turn.first_reply, "first_ungrounded": turn.first_ungrounded,
                           "ungrounded_after": turn.ungrounded},
            ))
        if turn.ungrounded:
            session.add(ToolLog(
                message_id=ctx.message_id, family_id=ctx.family_id, member_id=ctx.member_id, step=turn.llm_steps,
                tool_name="grounding", arguments={"numbers": turn.ungrounded}, result=None, status="ungrounded",
                model=model,
            ))
    session.add(Message(family_id=ctx.family_id, member_id=ctx.member_id, role="assistant", content=turn.reply,
                        tool_summary=_summary(turn.steps) or None, ended_by=turn.ended_by))
    session.commit()
    return turn
