import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def test_business_endpoints_use_default_student_dependency():
    from pathlib import Path

    api_dir = Path(__file__).parents[2] / "app" / "api"
    for filename in ("upload.py", "questions.py", "sheets.py", "profile.py"):
        source = (api_dir / filename).read_text(encoding="utf-8")
        assert "get_current_student" not in source
        assert "get_default_student" in source


def test_worker_checks_account_status_before_recognition():
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "app" / "tasks" / "process_image.py"
    ).read_text(encoding="utf-8")

    assert "Account.status == \"active\"" in source


class ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class ResultDB:
    def __init__(self, *values):
        self.values = list(values)
        self.queries = []

    async def execute(self, query):
        self.queries.append(query)
        return ScalarResult(self.values.pop(0))


def test_current_account_rejects_stale_token_version(monkeypatch):
    from app.api import deps
    from app.utils.jwt import TokenClaims

    monkeypatch.setattr(
        deps,
        "verify_token",
        lambda _token: TokenClaims(account_id="account-1", token_version=2),
    )
    db = ResultDB(
        SimpleNamespace(
            id="account-1",
            token_version=3,
            status="active",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deps.get_current_account(
                credentials=SimpleNamespace(credentials="token"),
                db=db,
            )
        )

    assert exc_info.value.status_code == 401


def test_current_account_rejects_missing_credentials_with_401():
    from app.api import deps

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deps.get_current_account(
                credentials=None,
                db=ResultDB(),
            )
        )

    assert exc_info.value.status_code == 401


def test_current_account_returns_active_account_context(monkeypatch):
    from app.api import deps
    from app.utils.jwt import TokenClaims

    monkeypatch.setattr(
        deps,
        "verify_token",
        lambda _token: TokenClaims(account_id="account-1", token_version=3),
    )
    db = ResultDB(
        SimpleNamespace(
            id="account-1",
            token_version=3,
            status="active",
        )
    )

    context = asyncio.run(
        deps.get_current_account(
            credentials=SimpleNamespace(credentials="token"),
            db=db,
        )
    )

    assert context.account_id == "account-1"
    assert context.token_version == 3


def test_current_account_rejects_pending_deletion(monkeypatch):
    from app.api import deps
    from app.utils.jwt import TokenClaims

    monkeypatch.setattr(
        deps,
        "verify_token",
        lambda _token: TokenClaims(account_id="account-1", token_version=3),
    )
    db = ResultDB(
        SimpleNamespace(
            id="account-1",
            token_version=3,
            status="pending_deletion",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            deps.get_current_account(
                credentials=SimpleNamespace(credentials="token"),
                db=db,
            )
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["code"] == "account_pending_deletion"


def test_deletion_retry_accepts_previous_normal_token_for_pending_account(
    monkeypatch,
):
    from app.api import deps
    from app.utils.jwt import NORMAL_TOKEN_SCOPE, TokenClaims

    dependency = getattr(deps, "get_deletion_account", None)
    assert dependency is not None

    monkeypatch.setattr(
        deps,
        "verify_token",
        lambda _token, required_scope=NORMAL_TOKEN_SCOPE: (
            TokenClaims(
                account_id="account-1",
                token_version=2,
                scope=NORMAL_TOKEN_SCOPE,
            )
            if required_scope == NORMAL_TOKEN_SCOPE
            else None
        ),
    )
    db = ResultDB(
        SimpleNamespace(
            id="account-1",
            token_version=3,
            status="pending_deletion",
        )
    )

    context = asyncio.run(
        dependency(
            credentials=SimpleNamespace(credentials="previous-normal-token"),
            db=db,
        )
    )

    assert context.account_id == "account-1"
    assert context.token_version == 3


def test_recovery_dependency_defers_expiry_to_account_deadline(monkeypatch):
    from app.api import deps
    from app.config import settings
    from app.utils.jwt import create_account_recovery_token

    monkeypatch.setattr(settings, "JWT_SECRET", "expired-recovery-secret")
    token = create_account_recovery_token(
        "account-1",
        3,
        datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    db = ResultDB(
        SimpleNamespace(
            id="account-1",
            token_version=3,
            status="pending_deletion",
        )
    )

    context = asyncio.run(
        deps.get_recovery_account(
            credentials=SimpleNamespace(credentials=token),
            db=db,
        )
    )

    assert context.account_id == "account-1"
    assert context.token_version == 3


def test_default_student_is_scoped_to_account():
    from app.api.deps import AccountContext, get_default_student

    student = SimpleNamespace(
        id="student-1",
        account_id="account-1",
        is_default=True,
    )
    db = ResultDB(student)

    result = asyncio.run(
        get_default_student(
            context=AccountContext(account_id="account-1", token_version=1),
            db=db,
        )
    )

    assert result is student
    compiled = str(db.queries[0])
    assert "students.account_id" in compiled
    assert "students.is_default" in compiled


def test_missing_default_student_returns_structured_error():
    from app.api.deps import AccountContext, get_default_student

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            get_default_student(
                context=AccountContext(
                    account_id="account-1",
                    token_version=1,
                ),
                db=ResultDB(None),
            )
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == {
        "code": "student_profile_required",
        "message": "请先完善当前学生资料",
    }
