"""Prints the two requests GP1 builds, without sending them (§3.1). Run by test_baseline_overrides.py.

It needs GP1's root venv and baseline/ as the working directory, which is why it is a script and not part of the test:
backend/.venv has neither GP1's imports nor its working directory.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import requests  # noqa: E402

from src.chat.engine import ChatEngine  # noqa: E402
from src.llm.parser import ExpenseParser  # noqa: E402

sent = []


class FakeReply:
    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def fake_post(url, **options):
    sent.append(options["json"])
    return FakeReply()


requests.post = fake_post
ExpenseParser("not-a-key")._call_llm("system", "user")
engine = ChatEngine.__new__(ChatEngine)  # __init__ would open ChromaDB and call Ollama; _call_llm needs neither
engine.api_key = "not-a-key"
engine._call_llm("system", "user")
print(json.dumps({"parser": sent[0], "engine": sent[1]}))
