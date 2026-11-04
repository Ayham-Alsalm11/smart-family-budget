"""Groq tool-calling smoke test (step 4b, §13).

    cd backend
    .venv\\Scripts\\python -m scripts.smoke_groq

Lists the Groq models, reports whether MODEL_MAIN, MODEL_SMALL and any Llama 3.1 8B are available,
and sends "دفعت 5000 بنزين" with one `add_expense` tool to MODEL_MAIN and MODEL_SMALL, once with the
default reasoning effort and once with reasoning_effort=low. Prints latency, token usage and the
rate-limit headers. The API key is read from the settings and never printed.
"""
import json
import sys
import time

import httpx

from app.db.categories import CATEGORY_NAMES
from app.settings import get_settings

GROQ_URL = "https://api.groq.com/openai/v1"
MESSAGE = "دفعت 5000 بنزين"
ADD_EXPENSE_TOOL = {
    "type": "function",
    "function": {
        "name": "add_expense",
        "description": "Record a household expense.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount paid, greater than 0."},
                "category": {"type": "string", "enum": CATEGORY_NAMES},
            },
            "required": ["amount", "category"],
        },
    },
}


def find_llama_8b(model_ids: list[str]) -> str | None:
    candidates = sorted(m for m in model_ids if "llama-3.1-8b" in m.lower())
    return candidates[0] if candidates else None


def rate_limits(response: httpx.Response) -> dict:
    return {k: v for k, v in response.headers.items() if k.startswith("x-ratelimit") or k == "retry-after"}


def try_tool_call(client: httpx.Client, model: str, reasoning_effort: str | None = None) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": MESSAGE}],
        "tools": [ADD_EXPENSE_TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 1024,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    started = time.perf_counter()
    response = client.post("/chat/completions", json=payload)
    seconds = round(time.perf_counter() - started, 2)
    result = {"model": model, "reasoning_effort": reasoning_effort or "default",
              "http_status": response.status_code, "seconds": seconds, "rate_limits": rate_limits(response)}
    if response.status_code != 200:
        result["error"] = response.text[:300]
        return result

    body = response.json()
    usage = body.get("usage") or {}
    result["prompt_tokens"] = usage.get("prompt_tokens")
    result["completion_tokens"] = usage.get("completion_tokens")
    result["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    message = body["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    result["tool_call"] = bool(calls)
    result["content"] = message.get("content")
    if calls:
        function = calls[0]["function"]
        result["tool_name"] = function["name"]
        try:
            result["arguments"] = json.loads(function["arguments"])
            result["arguments_valid_json"] = True
        except (TypeError, ValueError):
            result["arguments"] = function["arguments"]
            result["arguments_valid_json"] = False
    return result


def main() -> int:
    settings = get_settings()
    if not settings.groq_key_present:
        print("GROQ_API_KEY is not set in backend/.env.", file=sys.stderr)
        return 1
    headers = {"Authorization": f"Bearer {settings.groq_api_key.get_secret_value()}"}
    models = [settings.model_main, settings.model_small]

    with httpx.Client(base_url=GROQ_URL, headers=headers, timeout=30.0) as client:
        response = client.get("/models")
        response.raise_for_status()
        model_ids = sorted(m["id"] for m in response.json()["data"])
        print(f"Groq models ({len(model_ids)}): {', '.join(model_ids)}")
        print(f"Llama 3.1 8B: {find_llama_8b(model_ids)}")
        for name, model in zip(("MODEL_MAIN", "MODEL_SMALL"), models):
            print(f"{name} = {model} (listed: {model in model_ids})")

        ok = True
        for model in models:
            for effort in (None, "low"):
                result = try_tool_call(client, model, effort)
                print(json.dumps(result, ensure_ascii=False))
                if effort is None:
                    ok = ok and bool(result.get("tool_call")) and bool(result.get("arguments_valid_json"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
