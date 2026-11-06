"""The truncation check of §3.1, one level: does GP1's parser fit its reasoning and its JSON into max_tokens=200?

Started by eval/check_truncation.py, in GP1's environment from inside baseline/, once per reasoning level. Sentences go
through `parse_expense` only, because the risk is in the parser's limit; the answer step is not part of this check.
One line per sentence is written as soon as it ends. Exit code 2: the daily quota was reached.
"""
import argparse
import json
import os
import sys
from typing import Any

import gp1_http

DAILY_QUOTA_EXIT = 2


def check(parser: Any, log: gp1_http.RequestLog, sentence: str) -> dict[str, Any]:
    log.calls.clear()
    try:
        result = parser.parse_expense(sentence)
        kind, crash = result.get("type"), None
    except Exception as exc:  # A1: on a 429 in the last attempt GP1 raises outside its own try
        result, kind, crash = {}, None, f"{type(exc).__name__}: {exc}"
    last = log.calls[-1] if log.calls else {}
    return {
        "sentence": sentence, "type": kind, "amount": result.get("amount"), "category": result.get("category"),
        "valid_json": kind is not None and kind != "error", "crash": crash, "finish_reason": last.get("finish_reason"),
        "completion_tokens": last.get("completion_tokens"), "reasoning_tokens": last.get("reasoning_tokens"),
        "prompt_tokens": last.get("prompt_tokens"), "content_length": last.get("content_length"),
        "requests": len(log.calls), "statuses": [call.get("http_status") for call in log.calls],
        "total_tokens": sum(call.get("prompt_tokens", 0) + call.get("completion_tokens", 0) for call in log.calls),
    }


def main() -> int:
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arguments.add_argument("--sentences", required=True, help="a JSON list of sentences")
    arguments.add_argument("--out", required=True)
    args = arguments.parse_args()

    sys.path.insert(0, os.getcwd())
    log = gp1_http.RequestLog()
    log.install()
    from src.llm.parser import ExpenseParser  # noqa: E402

    parser = ExpenseParser(api_key=os.environ["GROQ_API_KEY"])
    with open(args.sentences, encoding="utf-8") as file:
        sentences = json.load(file)
    for sentence in sentences:
        item = check(parser, log, sentence)
        if log.daily_quota:
            print("Stopped: the daily quota was reached.", file=sys.stderr)
            return DAILY_QUOTA_EXIT
        with open(args.out, "a", encoding="utf-8") as file:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
