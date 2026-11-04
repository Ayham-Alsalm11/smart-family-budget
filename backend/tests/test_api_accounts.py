"""Registration, login, tokens, invites, members and family settings, through the API."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier

import jwt
import pytest
from sqlalchemy import func, select

from app.auth.passwords import hash_password
from app.auth.tokens import create_token
from app.db.models import Family, Invite, Member, PendingAction, utcnow
from app.services.accounts import INVITE_ALPHABET, deactivate_member, join_with_code
from app.services.errors import ConflictError
from app.services.expenses import add_expense, get_expense
from app.services.pending import execute_action

from api_support import (
    PASSWORD, SECRET, ask_to_change, assert_error, family_body, invite, join, join_body, make_api, register, spend,
)


@pytest.fixture
def api(tmp_path, monkeypatch):
    with make_api(tmp_path, monkeypatch) as client:
        yield client


def login(api, username, password=PASSWORD):
    return api.post("/auth/login", json={"username": username, "password": password})


def column(api, model, row_id, name):
    with api.app.state.session_factory() as session:
        return getattr(session.get(model, row_id), name)


def set_column(api, model, row_id, **values):
    with api.app.state.session_factory() as session:
        row = session.get(model, row_id)
        for name, value in values.items():
            setattr(row, name, value)
        session.commit()


# ---------- registration ----------

def test_register_family_makes_a_manager_and_logs_in(api):
    response = api.post("/auth/register-family", json=family_body("Abu.Sami_1"))
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer" and body["member"]["username"] == "abu.sami_1"
    assert body["member"]["role"] == "manager" and "password" not in response.text

    me = api.get("/me", headers={"Authorization": f"Bearer {body['access_token']}"}).json()
    assert me["member"]["name"] == "أبو سامي" and me["family"]["base_currency"] == "SYP_OLD"
    assert me["family"]["monthly_income"] == 500000
    assert login(api, "ABU.SAMI_1").status_code == 200, "login ignores the case of the username"


def test_register_default_currency_and_categories_exist(api):
    body = family_body("sami")
    del body["base_currency"]
    token = api.post("/auth/register-family", json=body).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert api.get("/me", headers=headers).json()["family"]["base_currency"] == "SYP_NEW"
    assert len(api.get("/budgets", headers=headers).json()["budgets"]) == 12, "start-up seeds the categories"


def test_username_taken_in_any_case_and_across_families(api):
    register(api, "sami")
    error = assert_error(api.post("/auth/register-family", json=family_body("SAMI")), 409, "conflict")
    assert error["fields"] == ["username"]
    with api.app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Family)) == 1, "the refused family was not created"


@pytest.mark.parametrize("username", ["ab", "a" * 31, "سامي", "has space", "dash-ed", "new\nline", "ok\n"])
def test_bad_usernames(api, username):
    error = assert_error(api.post("/auth/register-family", json=family_body(username)), 422, "invalid_args")
    assert error["fields"] == ["username"]


@pytest.mark.parametrize("username", ["abc", "a" * 30, "a.b_c9"])
def test_good_usernames(api, username):
    assert register(api, username).username == username


@pytest.mark.parametrize("name, status", [("س" * 50, 201), ("Sam", 201), ("س" * 51, 422), ("   ", 422)])
def test_display_name_any_language_up_to_50(api, name, status):
    response = api.post("/auth/register-family", json=family_body("sami", name=name))
    assert response.status_code == status


@pytest.mark.parametrize("password, status", [
    ("1234567", 422), ("12345678", 201), ("كلمة سر", 422), ("ك" * 36, 201), ("ك" * 37, 422), ("a" * 72, 201),
    ("a" * 73, 422),
])
def test_password_rules(api, password, status):
    response = api.post("/auth/register-family", json=family_body("sami", password=password))
    assert response.status_code == status, response.text
    if status == 422:
        assert response.json()["error"]["fields"] == ["password"]


def test_password_without_a_utf8_form(api):
    body = json.dumps(family_body("sami", password="bad\ud800password"))
    response = api.post("/auth/register-family", content=body, headers={"Content-Type": "application/json"})
    assert_error(response, 422, "invalid_args")
    assert_error(api.post("/auth/login", content='{"username": "sami", "password": "bad\\ud800password"}',
                          headers={"Content-Type": "application/json"}), 401, "unauthorized")


def test_long_arabic_password_error_is_clear(api):
    error = assert_error(api.post("/auth/register-family", json=family_body("sami", password="ك" * 40)), 422,
                         "invalid_args")
    assert "72" in error["message"] and "بايت" in error["message"]


def test_arabic_password_logs_in(api):
    register(api, "sami", password="ك" * 36)
    assert login(api, "sami", "ك" * 36).status_code == 200
    assert_error(login(api, "sami", "ك" * 37), 401, "unauthorized")


def test_missing_body_field_names_it(api):
    error = assert_error(api.post("/auth/login", json={"username": "sami"}), 422, "missing_fields")
    assert error["fields"] == ["password"]


def test_rejected_value_is_never_echoed(api):
    response = api.post("/auth/login", json={"username": "sami", "password": 12345678123})
    assert_error(response, 422, "invalid_args")
    assert "12345678123" not in response.text


@pytest.mark.parametrize("secret", [None, "x" * 31])
def test_login_paths_refuse_a_missing_or_short_secret(tmp_path, monkeypatch, secret):
    with make_api(tmp_path, monkeypatch, jwt_secret=secret) as api:
        error = assert_error(api.post("/auth/register-family", json=family_body("sami")), 503, "not_available")
        assert "JWT_SECRET" in error["message"]
        assert_error(login(api, "sami"), 503, "not_available")
        assert_error(api.post("/auth/join", json=join_body("AAAAAAAA", "sami")), 503, "not_available")
        with api.app.state.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(Family)) == 0


# ---------- login ----------

def test_one_message_for_unknown_user_and_wrong_password(api):
    register(api, "sami")
    unknown, wrong = login(api, "nobody"), login(api, "sami", "wrong-password")
    assert_error(unknown, 401, "unauthorized")
    assert_error(wrong, 401, "unauthorized")
    assert unknown.json() == wrong.json()


def test_login_rules_are_not_applied_to_old_passwords(api):
    """The seed's development passwords are shorter than the rule; the rule is for choosing one."""
    manager = register(api, "sami")
    set_column(api, Member, manager.id, password_hash=hash_password("1234"))
    assert login(api, "sami", "1234").status_code == 200


def test_lock_after_five_failures_even_with_the_right_password(api):
    manager = register(api, "sami")
    for _ in range(5):
        assert_error(login(api, "sami", "wrong-password"), 401, "unauthorized")
    assert column(api, Member, manager.id, "locked_until") is not None
    response = login(api, "sami")
    assert_error(response, 429, "locked")
    assert 0 < int(response.headers["retry-after"]) <= 300
    assert column(api, Member, manager.id, "failed_logins") == 5, "a locked attempt is not counted"


def test_lock_ends_after_its_time_and_counting_starts_again(api):
    manager = register(api, "sami")
    for _ in range(5):
        login(api, "sami", "wrong-password")
    set_column(api, Member, manager.id, locked_until=utcnow() - timedelta(seconds=1))
    assert_error(login(api, "sami", "wrong-password"), 401, "unauthorized")
    assert column(api, Member, manager.id, "failed_logins") == 1
    assert column(api, Member, manager.id, "locked_until") is None
    assert login(api, "sami").status_code == 200


def test_success_resets_the_counter(api):
    manager = register(api, "sami")
    for _ in range(2):
        for _ in range(4):
            login(api, "sami", "wrong-password")
        assert login(api, "sami").status_code == 200
        assert column(api, Member, manager.id, "failed_logins") == 0


def test_too_long_password_at_login_counts_as_wrong(api):
    register(api, "sami")
    assert_error(login(api, "sami", "a" * 100), 401, "unauthorized")


def test_stopped_member_cannot_login_with_the_same_message(api):
    manager = register(api, "sami")
    member = join(api, manager, "rana")
    assert api.delete(f"/members/{member.id}", headers=manager.headers).status_code == 200
    response = login(api, "rana")
    assert_error(response, 401, "unauthorized")
    assert response.json() == login(api, "nobody").json()


# ---------- tokens ----------

def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_no_token_and_malformed_tokens(api):
    assert_error(api.get("/me"), 401, "unauthorized")
    assert_error(api.get("/me", headers=bearer("not-a-token")), 401, "unauthorized")
    assert_error(api.get("/me", headers={"Authorization": "Basic abc"}), 401, "unauthorized")


def test_expired_token(api):
    manager = register(api, "sami")
    settings = api.app.state.settings
    old = create_token(settings, manager.id, now=utcnow() - timedelta(days=7, seconds=1))
    assert_error(api.get("/me", headers=bearer(old)), 401, "unauthorized")
    recent = create_token(settings, manager.id, now=utcnow() - timedelta(days=6))
    assert api.get("/me", headers=bearer(recent)).status_code == 200


def test_token_has_only_the_member_id(api):
    manager = register(api, "sami")
    claims = jwt.decode(manager.token, SECRET, algorithms=["HS256"])
    assert set(claims) == {"sub", "iat", "exp"} and claims["sub"] == str(manager.id)
    assert claims["exp"] - claims["iat"] == 7 * 24 * 3600


def test_altered_token(api):
    first, second = register(api, "sami"), register(api, "rana")
    header, _payload, signature = first.token.split(".")
    other_payload = second.token.split(".")[1]
    assert_error(api.get("/me", headers=bearer(f"{header}.{other_payload}.{signature}")), 401, "unauthorized")


@pytest.mark.filterwarnings("ignore:The HMAC key")
@pytest.mark.parametrize("key, algorithm", [("another-secret-of-the-right-length-32+", "HS256"), (SECRET, "HS512")])
def test_token_signed_differently(api, key, algorithm):
    manager = register(api, "sami")
    now = utcnow()
    forged = jwt.encode({"sub": str(manager.id), "iat": now, "exp": now + timedelta(days=1)}, key, algorithm=algorithm)
    assert_error(api.get("/me", headers=bearer(forged)), 401, "unauthorized")


def test_token_without_expiry(api):
    manager = register(api, "sami")
    forged = jwt.encode({"sub": str(manager.id), "iat": utcnow()}, SECRET, algorithm="HS256")
    assert_error(api.get("/me", headers=bearer(forged)), 401, "unauthorized")


def test_stopped_member_token_fails_at_once(api):
    manager = register(api, "sami")
    member = join(api, manager, "rana")
    assert api.get("/me", headers=member.headers).status_code == 200
    api.delete(f"/members/{member.id}", headers=manager.headers)
    assert_error(api.get("/me", headers=member.headers), 401, "unauthorized")


def test_token_of_a_member_that_no_longer_exists(api):
    token = create_token(api.app.state.settings, 999)
    assert_error(api.get("/me", headers=bearer(token)), 401, "unauthorized")


def test_role_is_read_from_the_database_not_the_token(api):
    manager = register(api, "sami")
    member = join(api, manager, "rana")
    assert_error(api.get("/members", headers=member.headers), 403, "forbidden")
    api.put(f"/members/{member.id}/role", json={"role": "manager"}, headers=manager.headers)
    assert api.get("/members", headers=member.headers).status_code == 200, "the same token, the new role"


# ---------- invites ----------

def test_invite_code_format_and_expiry(api):
    manager = register(api, "sami")
    created = invite(api, manager)
    assert len(created["code"]) == 8 and set(created["code"]) <= set(INVITE_ALPHABET)
    assert not set(INVITE_ALPHABET) & set("0O1IL") and created["role"] == "member"
    expires = column(api, Invite, created["id"], "expires_at")
    assert timedelta(days=6, hours=23) < expires - utcnow() <= timedelta(days=7)


def test_invite_default_role_without_a_body(api):
    manager = register(api, "sami")
    response = api.post("/invites", headers=manager.headers)
    assert response.status_code == 201 and response.json()["role"] == "member"


def test_invite_bad_role(api):
    manager = register(api, "sami")
    assert_error(api.post("/invites", json={"role": "owner"}, headers=manager.headers), 422, "invalid_args")


def test_member_cannot_touch_invites(api):
    manager = register(api, "sami")
    member = join(api, manager, "rana")
    open_invite = invite(api, manager)
    assert_error(api.post("/invites", json={"role": "member"}, headers=member.headers), 403, "forbidden")
    assert_error(api.get("/invites", headers=member.headers), 403, "forbidden")
    assert_error(api.delete(f"/invites/{open_invite['id']}", headers=member.headers), 403, "forbidden")


def test_code_is_single_use_and_accepted_in_any_case(api):
    manager = register(api, "sami")
    code = invite(api, manager)["code"]
    assert api.post("/auth/join", json=join_body(f" {code.lower()} ", "rana")).status_code == 201
    assert_error(api.post("/auth/join", json=join_body(code, "other")), 409, "conflict")
    with api.app.state.session_factory() as session:
        used = session.scalar(select(Invite).where(Invite.code == code))
        assert used.used_by is not None and used.used_at is not None


def test_unknown_code(api):
    assert_error(api.post("/auth/join", json=join_body("ZZZZZZZZ", "rana")), 404, "not_found")


def test_expired_code(api):
    manager = register(api, "sami")
    created = invite(api, manager)
    set_column(api, Invite, created["id"], expires_at=utcnow() - timedelta(seconds=1))
    assert_error(api.post("/auth/join", json=join_body(created["code"], "rana")), 409, "conflict")


def test_revoked_code(api):
    manager = register(api, "sami")
    created = invite(api, manager)
    assert api.delete(f"/invites/{created['id']}", headers=manager.headers).status_code == 200
    assert_error(api.post("/auth/join", json=join_body(created["code"], "rana")), 409, "conflict")
    assert_error(api.delete(f"/invites/{created['id']}", headers=manager.headers), 409, "conflict")


def test_failed_join_keeps_the_code(api):
    manager = register(api, "sami")
    code = invite(api, manager)["code"]
    assert_error(api.post("/auth/join", json=join_body(code, "sami")), 409, "conflict")
    assert api.post("/auth/join", json=join_body(code, "rana")).status_code == 201


@pytest.mark.parametrize("role", ["member", "manager"])
def test_role_and_family_come_from_the_invite(api, role):
    manager, other = register(api, "sami"), register(api, "omar")
    joined = api.post("/auth/join", json=join_body(
        invite(api, manager, role)["code"], "rana", role="manager" if role == "member" else "member",
        family_id=other.family_id, member_id=1)).json()
    assert joined["member"]["role"] == role
    with api.app.state.session_factory() as session:
        assert session.get(Member, joined["member"]["id"]).family_id == manager.family_id


def test_list_shows_open_invites_only(api):
    manager = register(api, "sami")
    used, revoked, expired, open_invite = (invite(api, manager) for _ in range(4))
    api.post("/auth/join", json=join_body(used["code"], "rana"))
    api.delete(f"/invites/{revoked['id']}", headers=manager.headers)
    set_column(api, Invite, expired["id"], expires_at=utcnow() - timedelta(seconds=1))
    listed = api.get("/invites", headers=manager.headers).json()["invites"]
    assert [i["id"] for i in listed] == [open_invite["id"]] and listed[0]["code"] == open_invite["code"]


# ---------- members ----------

def test_members_list_for_manager_without_secrets(api):
    manager = register(api, "sami")
    join(api, manager, "rana")
    response = api.get("/members", headers=manager.headers)
    assert [m["username"] for m in response.json()["members"]] == ["sami", "rana"]
    assert "password" not in response.text and "failed_logins" not in response.text


def test_last_manager_cannot_be_stopped_or_demoted(api):
    manager = register(api, "sami")
    assert_error(api.delete(f"/members/{manager.id}", headers=manager.headers), 409, "conflict")
    assert_error(api.put(f"/members/{manager.id}/role", json={"role": "member"}, headers=manager.headers), 409,
                 "conflict")
    assert column(api, Member, manager.id, "is_active") is True
    assert column(api, Member, manager.id, "role") == "manager"


def test_stopped_manager_does_not_count_as_a_manager(api):
    manager = register(api, "sami")
    second = join(api, manager, "rana", "manager")
    assert api.delete(f"/members/{second.id}", headers=manager.headers).status_code == 200
    assert_error(api.delete(f"/members/{manager.id}", headers=manager.headers), 409, "conflict")


def test_promotion_then_the_first_manager_can_step_down(api):
    manager = register(api, "sami")
    member = join(api, manager, "rana")
    promoted = api.put(f"/members/{member.id}/role", json={"role": "manager"}, headers=manager.headers)
    assert promoted.status_code == 200 and promoted.json()["role"] == "manager"
    demoted = api.put(f"/members/{manager.id}/role", json={"role": "member"}, headers=manager.headers)
    assert demoted.status_code == 200 and demoted.json()["role"] == "member"
    assert_error(api.get("/members", headers=manager.headers), 403, "forbidden")


def test_bad_role_and_stopped_targets(api):
    manager = register(api, "sami")
    member = join(api, manager, "rana")
    assert_error(api.put(f"/members/{member.id}/role", json={"role": "owner"}, headers=manager.headers), 422,
                 "invalid_args")
    api.delete(f"/members/{member.id}", headers=manager.headers)
    assert_error(api.delete(f"/members/{member.id}", headers=manager.headers), 409, "conflict")
    assert_error(api.put(f"/members/{member.id}/role", json={"role": "manager"}, headers=manager.headers), 409,
                 "conflict")


def test_stopping_a_member_keeps_expenses_and_cancels_pending_actions(api):
    manager = register(api, "sami")
    member = join(api, manager, "rana")
    expense = spend(api, member, 7000)
    action = ask_to_change(api, member, expense)
    api.delete(f"/members/{member.id}", headers=manager.headers)

    assert column(api, PendingAction, action, "status") == "cancelled"
    assert column(api, Member, member.id, "is_active") is False
    assert api.get("/summary", headers=manager.headers).json()["total"] == 7000
    rows = api.get("/expenses", params={"member": member.id}, headers=manager.headers).json()["expenses"]
    assert [(r["id"], r["member_id"]) for r in rows] == [(expense, member.id)]


def test_stopping_a_manager_revokes_their_open_invites(api):
    manager = register(api, "sami")
    second = join(api, manager, "rana", "manager")
    theirs, mine = invite(api, second), invite(api, manager)
    api.delete(f"/members/{second.id}", headers=manager.headers)
    assert column(api, Invite, theirs["id"], "revoked_at") is not None
    assert column(api, Invite, mine["id"], "revoked_at") is None
    assert_error(api.post("/auth/join", json=join_body(theirs["code"], "omar")), 409, "conflict")


# ---------- family ----------

def test_update_family_income_rate_and_name(api):
    manager = register(api, "sami")
    response = api.put("/family", json={"monthly_income": 750000, "usd_rate": 110.5, "name": "آل سامي"},
                       headers=manager.headers)
    assert response.status_code == 200
    family = api.get("/me", headers=manager.headers).json()["family"]
    assert (family["monthly_income"], family["usd_rate"], family["name"]) == (750000, 110.5, "آل سامي")


@pytest.mark.parametrize("body", [{}, {"usd_rate": 0}, {"usd_rate": -1}, {"usd_rate": None}, {"monthly_income": -1},
                                  {"base_currency": "EUR"}, {"name": ""}])
def test_update_family_bad_values(api, body):
    manager = register(api, "sami")
    assert api.put("/family", json=body, headers=manager.headers).status_code == 422


def test_base_currency_changes_only_without_expenses(api):
    manager = register(api, "sami", usd_rate=100)
    assert api.put("/family", json={"base_currency": "USD"}, headers=manager.headers).status_code == 200
    assert api.put("/family", json={"base_currency": "SYP_OLD"}, headers=manager.headers).status_code == 200
    spend(api, manager)
    error = assert_error(api.put("/family", json={"base_currency": "SYP_NEW", "monthly_income": 1},
                                 headers=manager.headers), 409, "conflict")
    assert error["fields"] == ["base_currency"]
    family = api.get("/me", headers=manager.headers).json()["family"]
    assert family["base_currency"] == "SYP_OLD" and family["monthly_income"] == 500000, "nothing changed"
    same = api.put("/family", json={"base_currency": "SYP_OLD"}, headers=manager.headers)
    assert same.status_code == 200, "naming the current currency is not a change"


def test_new_usd_rate_affects_new_expenses_only(api):
    manager = register(api, "sami", usd_rate=100)
    first = spend(api, manager, 10)
    api.put("/family", json={"usd_rate": 200}, headers=manager.headers)
    with api.app.state.session_factory() as session:
        usd = add_expense(session, manager.ctx, amount=1, category="فواتير", currency="USD")
        assert usd["amount_base"] == 20000 and get_expense(session, manager.ctx, first)["amount_base"] == 10


# ---------- parallel requests ----------

def in_parallel(*calls):
    """Run the calls at the same moment, each in its own thread and session. Returns who got through."""
    start = Barrier(len(calls))

    def run(call):
        start.wait()
        try:
            call()
            return True
        except ConflictError:
            return False

    with ThreadPoolExecutor(len(calls)) as pool:
        return list(pool.map(run, calls))


def test_two_managers_cannot_stop_each_other_at_once(api):
    first = register(api, "sami")
    second = join(api, first, "rana", "manager")

    def stop(actor, target):
        def call():
            with api.app.state.session_factory() as session:
                deactivate_member(session, actor.ctx, target.id)
        return call

    assert sorted(in_parallel(stop(first, second), stop(second, first))) == [False, True]
    with api.app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Member).where(Member.is_active.is_(True))) == 1


def test_one_code_cannot_be_used_twice_at_once(api):
    manager = register(api, "sami")
    code = invite(api, manager)["code"]

    def use(username):
        def call():
            with api.app.state.session_factory() as session:
                join_with_code(session, code=code, name="سامي", username=username, password=PASSWORD)
        return call

    assert sorted(in_parallel(use("rana"), use("omar"), use("lina"))) == [False, False, True]
    with api.app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Member)) == 2


def test_one_action_cannot_run_twice_at_once(api):
    manager = register(api, "sami")
    action = ask_to_change(api, manager, spend(api, manager, 1000), {"amount": 2000})

    def confirm():
        with api.app.state.session_factory() as session:
            execute_action(session, manager.ctx, action)

    assert sorted(in_parallel(confirm, confirm, confirm)) == [False, False, True]
