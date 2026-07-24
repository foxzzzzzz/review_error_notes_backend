import asyncio
from types import SimpleNamespace

import httpx
import pytest


class ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class LoginDB:
    def __init__(self, results):
        self.results = list(results)
        self.added = []
        self.flush_calls = 0

    async def execute(self, _query):
        return ScalarResult(self.results.pop(0))

    def add_all(self, values):
        self.added.extend(values)

    async def flush(self):
        self.flush_calls += 1
        for value in self.added:
            if getattr(value, "id", None) is None:
                value.id = f"generated-{value.__class__.__name__.lower()}"


def test_wechat_login_exchange_returns_only_identity_fields():
    from app.services.wechat import WeChatSession, exchange_login_code

    def handler(request):
        assert request.url.path == "/sns/jscode2session"
        assert request.url.params["js_code"] == "login-code"
        return httpx.Response(
            200,
            json={"openid": "openid-1", "unionid": "unionid-1", "session_key": "secret"},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await exchange_login_code(
                "login-code",
                "app-id",
                "app-secret",
                client=client,
            )

    assert asyncio.run(run()) == WeChatSession(
        openid="openid-1",
        unionid="unionid-1",
    )


def test_wechat_login_exchange_sanitizes_upstream_failure():
    from app.services.wechat import WeChatAPIError, exchange_login_code

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={"errcode": 40029, "errmsg": "invalid code detail"},
                )
            )
        ) as client:
            return await exchange_login_code(
                "bad-code",
                "app-id",
                "app-secret",
                client=client,
            )

    with pytest.raises(WeChatAPIError) as exc_info:
        asyncio.run(run())

    assert str(exc_info.value) == "Unable to complete WeChat login"
    assert "invalid code detail" not in str(exc_info.value)


def test_new_identity_creates_account_identity_and_default_student_without_commit():
    from app.models.account import Account
    from app.models.student import Student
    from app.models.wechat_identity import WeChatIdentity
    from app.services.account_login import login_with_identity

    db = LoginDB([None])
    account, student, created = asyncio.run(
        login_with_identity(
            db,
            appid="app-id",
            openid="openid-1",
            unionid="unionid-1",
        )
    )

    assert created is True
    assert isinstance(account, Account)
    assert isinstance(student, Student)
    assert student.account_id == account.id
    assert student.is_default is True
    assert student.grade is None
    assert student.semester is None
    identity = next(value for value in db.added if isinstance(value, WeChatIdentity))
    assert identity.account_id == account.id
    assert identity.appid == "app-id"
    assert identity.openid == "openid-1"
    assert db.flush_calls == 2
    assert not hasattr(db, "commit_calls")


def test_existing_identity_returns_same_account_and_default_student():
    from app.services.account_login import login_with_identity

    identity = SimpleNamespace(account_id="account-1", last_login_at=None)
    account = SimpleNamespace(id="account-1", status="active")
    student = SimpleNamespace(id="student-1", account_id="account-1", is_default=True)
    db = LoginDB([identity, account, student])

    result_account, result_student, created = asyncio.run(
        login_with_identity(db, appid="app-id", openid="openid-1")
    )

    assert created is False
    assert result_account is account
    assert result_student is student
    assert identity.last_login_at is not None


def test_versioned_jwt_round_trip_and_rejects_legacy_token(monkeypatch):
    from jose import jwt

    from app.config import settings
    from app.utils.jwt import create_token, verify_token

    monkeypatch.setattr(settings, "JWT_SECRET", "test-secret")
    token = create_token("account-1", 7)

    assert verify_token(token).account_id == "account-1"
    assert verify_token(token).token_version == 7

    legacy = jwt.encode(
        {"sub": "student-1"},
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert verify_token(legacy) is None
