"""Helpers for the API tests: accounts created through the API, and rows prepared through the services."""
from contextlib import contextmanager
from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.agent.context import ToolContext
from app.auth import passwords
from app.main import create_app
from app.services import expenses, pending
from app.settings import Settings

from conftest import TODAY, FakeEmbedder, FakeVectorStore

SECRET = "a-test-secret-that-is-long-enough-for-hs256"
PASSWORD = "correct-horse"


class ApiEmbedder(FakeEmbedder):
    def warmup(self) -> dict:
        return {"ok": True, "seconds": 0.0, "dimensions": 8}

    def status(self) -> dict:
        return {"ok": True, "model": "fake", "model_available": True}


@contextmanager
def make_api(tmp_path, monkeypatch, llm=None, **overrides):
    """A TestClient on a new temporary database. bcrypt runs at its lowest cost, so that the tests stay fast."""
    monkeypatch.setattr(passwords, "ROUNDS", 4)
    options = {"database_url": f"sqlite:///{(tmp_path / 'api.db').as_posix()}", "jwt_secret": SECRET,
               "agent_today": TODAY, "groq_api_key": None, **overrides}
    store = FakeVectorStore()
    app = create_app(Settings(_env_file=None, **options), embedder=ApiEmbedder(),
                     vector_store_factory=lambda _settings: store, llm_client=llm)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.store = store
        yield client


@dataclass
class Account:
    id: int
    family_id: int
    role: str
    username: str
    token: str

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    @property
    def ctx(self) -> ToolContext:
        return ToolContext(self.family_id, self.id, self.role, TODAY)


def _account(api, response) -> Account:
    body = response.json()
    profile = api.get("/me", headers={"Authorization": f"Bearer {body['access_token']}"}).json()
    return Account(body["member"]["id"], profile["family"]["id"], body["member"]["role"], body["member"]["username"],
                   body["access_token"])


def family_body(username: str, **extra) -> dict:
    return {"family_name": "عائلة الاختبار", "monthly_income": 500000, "base_currency": "SYP_OLD", "name": "أبو سامي",
            "username": username, "password": PASSWORD, **extra}


def register(api, username: str, **extra) -> Account:
    response = api.post("/auth/register-family", json=family_body(username, **extra))
    assert response.status_code == 201, response.text
    return _account(api, response)


def invite(api, manager: Account, role: str = "member") -> dict:
    response = api.post("/invites", json={"role": role}, headers=manager.headers)
    assert response.status_code == 201, response.text
    return response.json()


def join_body(code: str, username: str, **extra) -> dict:
    return {"code": code, "name": "سامي", "username": username, "password": PASSWORD, **extra}


def join(api, manager: Account, username: str, role: str = "member") -> Account:
    response = api.post("/auth/join", json=join_body(invite(api, manager, role)["code"], username))
    assert response.status_code == 201, response.text
    return _account(api, response)


def spend(api, who: Account, amount=1000, category="أكل وشرب", description="", expense_date=None) -> int:
    with api.app.state.session_factory() as session:
        return expenses.add_expense(session, who.ctx, amount=amount, category=category, description=description,
                                    expense_date=expense_date)["id"]


def ask_to_change(api, who: Account, expense_id: int, changes: dict | None = None) -> int:
    """A pending edit (with `changes`) or deletion, as the tools create it. Returns its id."""
    tool = "edit_expense" if changes else "delete_expense"
    with api.app.state.session_factory() as session:
        return pending.request_change(session, who.ctx, tool, expense_id, changes)["confirmation_id"]


def assert_error(response, status: int, code: str) -> dict:
    """The one error shape of §8."""
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and code == body["error"]["code"]
    assert set(body["error"]) <= {"code", "message", "fields"} and body["error"]["message"]
    return body["error"]
