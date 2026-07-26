import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from jose import jwt


PHONE = "13800138000"
FINGERPRINT = "a" * 64


class RecoveryDB:
    def __init__(self):
        self.added = []
        self.deleted = []
        self.flush_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        self.flush_calls += 1

    async def delete(self, value):
        self.deleted.append(value)

    async def commit(self):
        self.commit_calls += 1

    async def rollback(self):
        self.rollback_calls += 1


def _patch_binding_queries(
    monkeypatch,
    recovery,
    *,
    current,
    target=None,
    current_has_data=False,
):
    async def lock_phone(_db, fingerprint):
        assert fingerprint == FINGERPRINT

    async def find_target(_db, fingerprint):
        assert fingerprint == FINGERPRINT
        return None if target is None else target.id

    async def lock_accounts(_db, account_ids):
        expected = {str(current.id)}
        accounts = {str(current.id): current}
        if target is not None:
            expected.add(str(target.id))
            accounts[str(target.id)] = target
        assert {str(value) for value in account_ids} == expected
        return accounts

    async def has_data(_db, account_id):
        assert str(account_id) == str(current.id)
        return current_has_data

    monkeypatch.setattr(recovery, "_lock_phone_fingerprint", lock_phone)
    monkeypatch.setattr(recovery, "_find_account_id_by_phone", find_target)
    monkeypatch.setattr(recovery, "_lock_accounts", lock_accounts)
    monkeypatch.setattr(recovery, "account_has_business_data", has_data)
    monkeypatch.setattr(recovery, "fingerprint_phone", lambda _phone: FINGERPRINT)


def test_unclaimed_phone_binds_to_current_account(monkeypatch):
    from app.services import account_recovery as recovery

    current = SimpleNamespace(
        id="current-account",
        phone_ciphertext=None,
        phone_fingerprint=None,
        phone_bound_at=None,
        token_version=1,
        status="active",
    )
    _patch_binding_queries(monkeypatch, recovery, current=current)
    monkeypatch.setattr(recovery, "encrypt_phone", lambda _phone: "encrypted-phone")

    result = asyncio.run(
        recovery.bind_phone_to_account(RecoveryDB(), current.id, PHONE)
    )

    assert result.status == "bound"
    assert result.phone_masked == "138****8000"
    assert current.phone_ciphertext == "encrypted-phone"
    assert current.phone_fingerprint == FINGERPRINT
    assert current.phone_bound_at is not None


def test_same_account_phone_rebind_is_idempotent(monkeypatch):
    from app.services import account_recovery as recovery

    original_bound_at = object()
    current = SimpleNamespace(
        id="current-account",
        phone_ciphertext="original-ciphertext",
        phone_fingerprint=FINGERPRINT,
        phone_bound_at=original_bound_at,
        token_version=1,
        status="active",
    )
    _patch_binding_queries(
        monkeypatch,
        recovery,
        current=current,
        target=current,
    )
    monkeypatch.setattr(
        recovery,
        "encrypt_phone",
        lambda _phone: pytest.fail("idempotent rebind must not re-encrypt"),
    )

    result = asyncio.run(
        recovery.bind_phone_to_account(RecoveryDB(), current.id, PHONE)
    )

    assert result.status == "bound"
    assert current.phone_ciphertext == "original-ciphertext"
    assert current.phone_bound_at is original_bound_at


def test_conflicting_phone_offers_recovery_when_current_account_is_empty(
    monkeypatch,
):
    from app.services import account_recovery as recovery

    current = SimpleNamespace(
        id="current-account",
        phone_ciphertext=None,
        phone_fingerprint=None,
        phone_bound_at=None,
        token_version=2,
        status="active",
    )
    target = SimpleNamespace(
        id="target-account",
        phone_ciphertext="target-ciphertext",
        phone_fingerprint=FINGERPRINT,
        phone_bound_at=object(),
        token_version=7,
        status="active",
    )
    _patch_binding_queries(
        monkeypatch,
        recovery,
        current=current,
        target=target,
        current_has_data=False,
    )

    with pytest.raises(recovery.AccountRecoveryAvailable) as exc_info:
        asyncio.run(
            recovery.bind_phone_to_account(RecoveryDB(), current.id, PHONE)
        )

    claims = recovery.verify_account_recovery_token(
        exc_info.value.recovery_token
    )
    assert claims.current_account_id == current.id
    assert claims.target_account_id == target.id
    assert claims.phone_fingerprint == FINGERPRINT
    assert claims.current_token_version == 2
    assert claims.target_token_version == 7


def test_conflicting_populated_accounts_require_manual_merge(monkeypatch):
    from app.services import account_recovery as recovery

    current = SimpleNamespace(
        id="current-account",
        phone_ciphertext=None,
        phone_fingerprint=None,
        phone_bound_at=None,
        token_version=2,
        status="active",
    )
    target = SimpleNamespace(
        id="target-account",
        phone_ciphertext="target-ciphertext",
        phone_fingerprint=FINGERPRINT,
        phone_bound_at=object(),
        token_version=7,
        status="active",
    )
    _patch_binding_queries(
        monkeypatch,
        recovery,
        current=current,
        target=target,
        current_has_data=True,
    )

    db = RecoveryDB()
    with pytest.raises(recovery.AccountMergeRequired) as exc_info:
        asyncio.run(
            recovery.bind_phone_to_account(db, current.id, PHONE)
        )

    assert exc_info.value.support_reference.startswith("AR-")
    assert len(exc_info.value.support_reference) == 15
    assert len(db.added) == 1
    conflict = db.added[0]
    assert conflict.support_reference == exc_info.value.support_reference
    assert str(conflict.current_account_id) == current.id
    assert str(conflict.target_account_id) == target.id
    assert conflict.status == "open"
    assert current.phone_fingerprint is None
    assert target.phone_fingerprint == FINGERPRINT


def test_pending_deletion_target_is_not_offered_as_empty_account_recovery(
    monkeypatch,
):
    from app.services import account_recovery as recovery

    current = SimpleNamespace(
        id="current-account",
        phone_ciphertext=None,
        phone_fingerprint=None,
        phone_bound_at=None,
        token_version=2,
        status="active",
    )
    target = SimpleNamespace(
        id="target-account",
        phone_ciphertext="target-ciphertext",
        phone_fingerprint=FINGERPRINT,
        phone_bound_at=object(),
        token_version=7,
        status="pending_deletion",
    )
    _patch_binding_queries(
        monkeypatch,
        recovery,
        current=current,
        target=target,
    )

    with pytest.raises(recovery.TargetAccountPendingDeletion):
        asyncio.run(
            recovery.bind_phone_to_account(
                RecoveryDB(),
                current.id,
                PHONE,
            )
        )


def test_confirmed_recovery_moves_identity_and_deletes_empty_placeholder(
    monkeypatch,
):
    from app.services import account_recovery as recovery

    current = SimpleNamespace(
        id="current-account",
        token_version=2,
        status="active",
        avatar_object_key="/avatars/current-avatar.jpg",
    )
    target = SimpleNamespace(
        id="target-account",
        token_version=7,
        status="active",
        phone_fingerprint=FINGERPRINT,
    )
    identity = SimpleNamespace(id="identity-1", account_id=current.id)
    current_student = SimpleNamespace(
        id="current-student",
        account_id=current.id,
        is_default=True,
    )
    target_student = SimpleNamespace(
        id="target-student",
        account_id=target.id,
        is_default=True,
    )
    token = recovery.create_account_recovery_token(
        current_account=current,
        target_account=target,
        phone_fingerprint=FINGERPRINT,
    )
    events = []

    async def lock_phone(_db, fingerprint):
        assert fingerprint == FINGERPRINT

    async def lock_accounts(_db, account_ids):
        events.append("lock_accounts")
        assert {str(value) for value in account_ids} == {
            current.id,
            target.id,
        }
        return {
            current.id: current,
            target.id: target,
        }

    async def has_data(_db, account_id):
        events.append("has_data")
        assert account_id == current.id
        return False

    async def lock_identities(_db, account_id):
        events.append("lock_identities")
        assert account_id == current.id
        return [identity]

    async def lock_students(_db, account_id):
        events.append(f"lock_students:{account_id}")
        if account_id == current.id:
            return [current_student]
        assert account_id == target.id
        return [target_student]

    monkeypatch.setattr(recovery, "_lock_phone_fingerprint", lock_phone)
    monkeypatch.setattr(recovery, "_lock_accounts", lock_accounts)
    monkeypatch.setattr(recovery, "account_has_business_data", has_data)
    monkeypatch.setattr(recovery, "_lock_identities", lock_identities)
    monkeypatch.setattr(recovery, "_lock_students", lock_students)
    db = RecoveryDB()

    result = asyncio.run(
        recovery.recover_empty_account(db, token)
    )

    assert result.account is target
    assert result.student is target_student
    assert result.cleanup_job is db.added[0]
    assert result.cleanup_job.storage_kind == "avatar"
    assert result.cleanup_job.object_path == "/avatars/current-avatar.jpg"
    assert identity.account_id == target.id
    assert target.token_version == 8
    assert current.token_version == 3
    assert db.deleted == [current]
    assert db.flush_calls == 2
    assert events.index("has_data") > events.index(
        f"lock_students:{current.id}"
    )


def test_business_data_query_covers_all_phase_four_ownership_sources():
    from app.services.account_recovery import account_has_business_data

    class ScalarDB:
        def __init__(self):
            self.statement = None

        async def scalar(self, statement):
            self.statement = statement
            return False

    db = ScalarDB()

    assert asyncio.run(account_has_business_data(db, "account-1")) is False
    sql = str(db.statement)
    for table_name in (
        "wrong_images",
        "wrong_questions",
        "practice_sheets",
        "practice_attempts",
        "students",
    ):
        assert table_name in sql


def test_account_recovery_token_expiry_is_externally_configured(monkeypatch):
    from app.config import Settings
    from app.services import account_recovery as recovery

    monkeypatch.delenv("ACCOUNT_RECOVERY_TOKEN_EXPIRE_MINUTES", raising=False)

    settings = Settings(_env_file=None)
    now = datetime(2026, 7, 25, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        recovery.settings,
        "ACCOUNT_RECOVERY_TOKEN_EXPIRE_MINUTES",
        10,
    )
    monkeypatch.setattr(recovery, "_utc_now", lambda: now)
    token = recovery.create_account_recovery_token(
        current_account=SimpleNamespace(id="current", token_version=1),
        target_account=SimpleNamespace(id="target", token_version=2),
        phone_fingerprint=FINGERPRINT,
    )
    claims = jwt.get_unverified_claims(token)

    assert settings.ACCOUNT_RECOVERY_TOKEN_EXPIRE_MINUTES == 10
    assert claims["exp"] == int(now.timestamp()) + 10 * 60


def test_compose_injects_account_recovery_token_expiry():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "ACCOUNT_RECOVERY_TOKEN_EXPIRE_MINUTES:" in compose


def test_bind_phone_endpoint_commits_success(monkeypatch):
    from app.api import auth as auth_api
    from app.api.deps import AccountContext
    from app.schemas.auth import BindPhoneRequest
    from app.services.account_recovery import BindPhoneResult

    async def get_phone(_code, _appid, _secret):
        return PHONE

    async def bind(_db, account_id, phone):
        assert account_id == "current-account"
        assert phone == PHONE
        return BindPhoneResult(status="bound", phone_masked="138****8000")

    monkeypatch.setattr(auth_api, "get_phone_number", get_phone)
    monkeypatch.setattr(auth_api, "bind_phone_to_account", bind)
    db = RecoveryDB()

    result = asyncio.run(
        auth_api.bind_phone(
            BindPhoneRequest(code="wechat-phone-code"),
            context=AccountContext(
                account_id="current-account",
                token_version=1,
            ),
            db=db,
        )
    )

    assert result.model_dump() == {
        "status": "bound",
        "phone_masked": "138****8000",
    }
    assert db.commit_calls == 1
    assert db.rollback_calls == 0


def test_bind_phone_endpoint_returns_recovery_conflict(monkeypatch):
    from app.api import auth as auth_api
    from app.api.deps import AccountContext
    from app.schemas.auth import BindPhoneRequest

    async def get_phone(_code, _appid, _secret):
        return PHONE

    async def bind(_db, _account_id, _phone):
        raise auth_api.AccountRecoveryAvailable(
            recovery_token="recovery-token",
            phone_masked="138****8000",
        )

    monkeypatch.setattr(auth_api, "get_phone_number", get_phone)
    monkeypatch.setattr(auth_api, "bind_phone_to_account", bind)
    db = RecoveryDB()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            auth_api.bind_phone(
                BindPhoneRequest(code="wechat-phone-code"),
                context=AccountContext(
                    account_id="current-account",
                    token_version=1,
                ),
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "code": "account_recovery_available",
        "message": "该手机号关联了已有账户，是否恢复原账户？",
        "phone_masked": "138****8000",
        "recovery_token": "recovery-token",
    }
    assert db.commit_calls == 0
    assert db.rollback_calls == 1


def test_bind_phone_endpoint_returns_manual_merge_reference(monkeypatch):
    from app.api import auth as auth_api
    from app.api.deps import AccountContext
    from app.schemas.auth import BindPhoneRequest

    async def get_phone(_code, _appid, _secret):
        return PHONE

    async def bind(_db, _account_id, _phone):
        raise auth_api.AccountMergeRequired("AR-123456789ABC")

    monkeypatch.setattr(auth_api, "get_phone_number", get_phone)
    monkeypatch.setattr(auth_api, "bind_phone_to_account", bind)
    db = RecoveryDB()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            auth_api.bind_phone(
                BindPhoneRequest(code="wechat-phone-code"),
                context=AccountContext(
                    account_id="current-account",
                    token_version=1,
                ),
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "account_merge_required"
    assert exc_info.value.detail["support_reference"] == "AR-123456789ABC"
    assert db.commit_calls == 1
    assert db.rollback_calls == 0


def test_bind_phone_endpoint_rejects_pending_deletion_target(monkeypatch):
    from app.api import auth as auth_api
    from app.api.deps import AccountContext
    from app.schemas.auth import BindPhoneRequest

    async def get_phone(_code, _appid, _secret):
        return PHONE

    async def bind(_db, _account_id, _phone):
        raise auth_api.TargetAccountPendingDeletion()

    monkeypatch.setattr(auth_api, "get_phone_number", get_phone)
    monkeypatch.setattr(auth_api, "bind_phone_to_account", bind)
    db = RecoveryDB()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            auth_api.bind_phone(
                BindPhoneRequest(code="wechat-phone-code"),
                context=AccountContext(
                    account_id="current-account",
                    token_version=1,
                ),
                db=db,
            )
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "account_pending_deletion"
    assert db.commit_calls == 0
    assert db.rollback_calls == 1


def test_recover_account_endpoint_returns_target_session(monkeypatch):
    from app.api import auth as auth_api
    from app.schemas.auth import RecoverAccountRequest
    from app.services.account_recovery import AccountRecoveryResult

    account_id = uuid4()
    student_id = uuid4()
    account = SimpleNamespace(
        id=account_id,
        token_version=8,
        profile_prompted_at=None,
        profile_completed_at=None,
        status="active",
    )
    student = SimpleNamespace(
        id=student_id,
        grade=1,
        semester=2,
    )

    async def recover(_db, token):
        assert token == "recovery-token"
        return AccountRecoveryResult(
            account=account,
            student=student,
            cleanup_job=SimpleNamespace(id="cleanup-job"),
        )

    cleanup_calls = []

    async def cleanup(_db, job_id):
        cleanup_calls.append(job_id)
        return True

    monkeypatch.setattr(auth_api, "recover_empty_account", recover)
    monkeypatch.setattr(auth_api, "attempt_file_cleanup", cleanup)
    db = RecoveryDB()

    result = asyncio.run(
        auth_api.recover_account(
            RecoverAccountRequest(recovery_token="recovery-token"),
            db=db,
        )
    )

    assert result.account_id == account_id
    assert result.student_id == student_id
    assert result.is_new_account is False
    assert result.student_profile_required is False
    assert db.commit_calls == 1
    assert db.rollback_calls == 0
    assert cleanup_calls == ["cleanup-job"]
