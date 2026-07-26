import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


class DeletionDB:
    def __init__(self):
        self.commit_calls = 0
        self.rollback_calls = 0

    async def commit(self):
        self.commit_calls += 1

    async def rollback(self):
        self.rollback_calls += 1


def _account(*, status="active", token_version=4, due_at=None):
    return SimpleNamespace(
        id=uuid4(),
        status=status,
        token_version=token_version,
        deletion_requested_at=None,
        deletion_due_at=due_at,
        profile_prompted_at=None,
        profile_completed_at=None,
    )


def _student(account):
    return SimpleNamespace(
        id=uuid4(),
        account_id=account.id,
        grade=1,
        semester=2,
    )


def test_account_deletion_retention_is_externally_configured(monkeypatch):
    from app.config import Settings

    monkeypatch.delenv("ACCOUNT_DELETION_RETENTION_DAYS", raising=False)

    settings = Settings(_env_file=None)
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert settings.ACCOUNT_DELETION_RETENTION_DAYS == 30
    assert "ACCOUNT_DELETION_RETENTION_DAYS:" in compose
    assert "ACCOUNT_DELETION_RETENTION_DAYS=30" in env_example


def test_deletion_request_requires_a_fresh_login_code():
    from app.schemas.account import AccountDeletionRequest

    with pytest.raises(ValidationError):
        AccountDeletionRequest(code="")


def test_scoped_tokens_cannot_be_used_for_the_wrong_purpose(monkeypatch):
    from app.config import settings
    from app.utils.jwt import (
        ACCOUNT_RECOVERY_TOKEN_SCOPE,
        NORMAL_TOKEN_SCOPE,
        create_account_recovery_token,
        create_token,
        verify_token,
    )

    monkeypatch.setattr(settings, "JWT_SECRET", "deletion-test-secret")
    due_at = datetime.now(timezone.utc) + timedelta(days=30)
    normal_token = create_token("account-1", 3)
    recovery_token = create_account_recovery_token(
        "account-1",
        4,
        due_at,
    )

    assert verify_token(normal_token).scope == NORMAL_TOKEN_SCOPE
    assert verify_token(
        recovery_token,
        required_scope=ACCOUNT_RECOVERY_TOKEN_SCOPE,
    ).scope == ACCOUNT_RECOVERY_TOKEN_SCOPE
    assert verify_token(recovery_token) is None
    assert verify_token(
        normal_token,
        required_scope=ACCOUNT_RECOVERY_TOKEN_SCOPE,
    ) is None


def test_deletion_requires_the_fresh_identity_to_belong_to_current_account(
    monkeypatch,
):
    from app.services import account_deletion

    account = _account()

    async def lock_account(_db, account_id):
        assert account_id == account.id
        return account

    async def identity_matches(_db, account_id, appid, openid):
        assert account_id == account.id
        assert appid == "app-id"
        assert openid == "another-openid"
        return False

    monkeypatch.setattr(account_deletion, "_lock_account", lock_account)
    monkeypatch.setattr(
        account_deletion,
        "_identity_belongs_to_account",
        identity_matches,
    )

    with pytest.raises(account_deletion.InvalidDeletionIdentity):
        asyncio.run(
            account_deletion.request_account_deletion(
                DeletionDB(),
                account.id,
                appid="app-id",
                openid="another-openid",
            )
        )

    assert account.status == "active"
    assert account.token_version == 4


def test_successful_deletion_invalidates_old_tokens_and_preserves_data(
    monkeypatch,
):
    from app.config import settings
    from app.services import account_deletion

    account = _account()
    now = datetime(2026, 7, 26, 2, 0)

    async def lock_account(_db, _account_id):
        return account

    async def identity_matches(_db, _account_id, _appid, _openid):
        return True

    monkeypatch.setattr(account_deletion, "_lock_account", lock_account)
    monkeypatch.setattr(
        account_deletion,
        "_identity_belongs_to_account",
        identity_matches,
    )
    monkeypatch.setattr(settings, "ACCOUNT_DELETION_RETENTION_DAYS", 30)

    result = asyncio.run(
        account_deletion.request_account_deletion(
            DeletionDB(),
            account.id,
            appid="app-id",
            openid="openid-1",
            now=now,
        )
    )

    assert result is account
    assert account.status == "pending_deletion"
    assert account.deletion_requested_at == now
    assert account.deletion_due_at == now + timedelta(days=30)
    assert account.token_version == 5


def test_repeated_deletion_is_idempotent(monkeypatch):
    from app.services import account_deletion

    requested_at = datetime(2026, 7, 26, 2, 0)
    due_at = requested_at + timedelta(days=30)
    account = _account(
        status="pending_deletion",
        token_version=5,
        due_at=due_at,
    )
    account.deletion_requested_at = requested_at

    async def lock_account(_db, _account_id):
        return account

    async def identity_matches(_db, _account_id, _appid, _openid):
        return True

    monkeypatch.setattr(account_deletion, "_lock_account", lock_account)
    monkeypatch.setattr(
        account_deletion,
        "_identity_belongs_to_account",
        identity_matches,
    )

    result = asyncio.run(
        account_deletion.request_account_deletion(
            DeletionDB(),
            account.id,
            appid="app-id",
            openid="openid-1",
            now=requested_at + timedelta(minutes=5),
        )
    )

    assert result is account
    assert account.deletion_requested_at == requested_at
    assert account.deletion_due_at == due_at
    assert account.token_version == 5


def test_recovery_within_retention_reactivates_account(monkeypatch):
    from app.services import account_deletion

    now = datetime(2026, 7, 26, 2, 0)
    account = _account(
        status="pending_deletion",
        token_version=5,
        due_at=now + timedelta(days=1),
    )
    account.deletion_requested_at = now - timedelta(days=1)
    student = _student(account)

    async def lock_account(_db, _account_id):
        return account

    async def get_default_student(_db, account_id):
        assert account_id == account.id
        return student

    monkeypatch.setattr(account_deletion, "_lock_account", lock_account)
    monkeypatch.setattr(
        account_deletion,
        "_get_default_student",
        get_default_student,
    )

    result_account, result_student = asyncio.run(
        account_deletion.recover_account_deletion(
            DeletionDB(),
            account.id,
            token_version=5,
            now=now,
        )
    )

    assert result_account is account
    assert result_student is student
    assert account.status == "active"
    assert account.deletion_requested_at is None
    assert account.deletion_due_at is None
    assert account.token_version == 6


def test_recovery_after_due_time_is_rejected(monkeypatch):
    from app.services import account_deletion

    now = datetime(2026, 7, 26, 2, 0)
    account = _account(
        status="pending_deletion",
        token_version=5,
        due_at=now,
    )

    async def lock_account(_db, _account_id):
        return account

    monkeypatch.setattr(account_deletion, "_lock_account", lock_account)

    with pytest.raises(account_deletion.AccountDeletionExpired):
        asyncio.run(
            account_deletion.recover_account_deletion(
                DeletionDB(),
                account.id,
                token_version=5,
                now=now,
            )
        )

    assert account.status == "pending_deletion"
    assert account.token_version == 5


def test_pending_login_returns_only_recovery_token(monkeypatch):
    from app.api import auth as auth_api
    from app.config import settings
    from app.utils.jwt import ACCOUNT_RECOVERY_TOKEN_SCOPE, verify_token

    monkeypatch.setattr(settings, "JWT_SECRET", "pending-login-secret")
    due_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
    account = _account(
        status="pending_deletion",
        token_version=5,
        due_at=due_at,
    )
    student = _student(account)

    response = auth_api._login_response(account, student, False)

    assert response.token is None
    assert response.recovery_token
    assert response.account_status == "pending_deletion"
    assert response.deletion_due_at == due_at.replace(tzinfo=timezone.utc)
    assert verify_token(
        response.recovery_token,
        required_scope=ACCOUNT_RECOVERY_TOKEN_SCOPE,
    ).account_id == str(account.id)


def test_pending_login_after_due_time_returns_expired_error():
    from app.api import auth as auth_api

    account = _account(
        status="pending_deletion",
        token_version=5,
        due_at=datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(seconds=1),
    )

    with pytest.raises(HTTPException) as exc_info:
        auth_api._login_response(account, _student(account), False)

    assert exc_info.value.status_code == 410
    assert exc_info.value.detail["code"] == "account_deletion_expired"


def test_recovery_dependency_rejects_normal_token(monkeypatch):
    from app.api import deps
    from app.config import settings
    from app.utils.jwt import create_token

    monkeypatch.setattr(settings, "JWT_SECRET", "recovery-dependency-secret")
    token = create_token("account-1", 3)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deps.get_recovery_account(
                credentials=SimpleNamespace(credentials=token),
                db=DeletionDB(),
            )
        )

    assert exc_info.value.status_code == 401


def test_logout_is_authenticated_and_idempotent():
    from app.api.account import logout
    from app.api.deps import AccountContext

    result = asyncio.run(
        logout(
            context=AccountContext(
                account_id="account-1",
                token_version=3,
            )
        )
    )

    assert result.model_dump() == {"ok": True}
