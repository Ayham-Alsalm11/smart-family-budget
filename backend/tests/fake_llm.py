"""A scripted stand-in for GroqClient (§6.3): returns prepared responses in order and records every request."""
import copy
import itertools
import json

from app.llm.client import Attempt, LLMError, LLMResponse, ToolCall, Usage

_ids = itertools.count(1)


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(f"call_{next(_ids)}", name, arguments, json.dumps(arguments, ensure_ascii=False))


def respond(*calls: ToolCall, content: str | None = None, usage: tuple[int, int, int] = (100, 20, 5)) -> LLMResponse:
    assistant = {"role": "assistant", "content": content}
    if calls:
        assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": c.raw_arguments}} for c in calls]
    return LLMResponse("fake-model", content, list(calls), Usage(*usage), 5, 1, assistant)


class FakeLLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict]] = []
        self.tools: list | None = None

    def chat(self, model, messages, tools=None, *, on_attempt=None, **_options) -> LLMResponse:
        self.requests.append(copy.deepcopy(messages))
        self.tools = tools
        if not self.responses:
            raise AssertionError("the fake model ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, LLMError):
            if on_attempt:
                on_attempt(Attempt(model, 1, item.kind, item.http_status, 5, None, str(item),
                                   retry_after=getattr(item, "retry_after", None)))
            raise item
        if on_attempt:
            on_attempt(Attempt(model, 1, "ok", 200, item.latency_ms, item.usage))
        return item
