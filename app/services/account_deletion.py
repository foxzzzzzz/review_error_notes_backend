from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.config import settings
from app.models.account import Account
from app.models.student import Student
from app.models.wechat_identity import WeChatIdentity


class InvalidDeletionIdentity(RuntimeError):
    pass


class AccountDeletionUnavailable(RuntimeError):
    pass


class AccountDeletionExpired(RuntimeError):
    pass


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _lock_account(db, account_id):
    return await db.scalar(
        select(Account)
        .where(Account.id == account_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )


async def _identity_belongs_to_account(
    db,
    account_id,
    appid: str,
    openid: str,
) -> bool:
    identity_id = await db.scalar(
        select(WeChatIdentity.id).where(
            WeChatIdentity.account_id == account_id,
            WeChatIdentity.appid == appid,
            WeChatIdentity.openid == openid,
        )
    )
    return identity_id is not None


async def _get_default_student(db, account_id):
    return await db.scalar(
        select(Student).where(
            Student.account_id == account_id,
            Student.is_default.is_(True),
        )
    )


async def request_account_deletion(
    db,
    account_id,
    *,
    appid: str,
    openid: str,
    now: datetime | None = None,
) -> Account:
    account = await _lock_account(db, account_id)
    if account is None:
        raise AccountDeletionUnavailable()
    if not await _identity_belongs_to_account(
        db,
        account.id,
        appid,
        openid,
    ):
        raise InvalidDeletionIdentity()
    if account.status == "pending_deletion":
        return account
    if account.status != "active":
        raise AccountDeletionUnavailable()

    requested_at = now or _utc_now_naive()
    account.status = "pending_deletion"
    account.deletion_requested_at = requested_at
    account.deletion_due_at = requested_at + timedelta(
        days=settings.ACCOUNT_DELETION_RETENTION_DAYS
    )
    account.token_version += 1
    return account


async def recover_account_deletion(
    db,
    account_id,
    *,
    token_version: int,
    now: datetime | None = None,
) -> tuple[Account, Student]:
    account = await _lock_account(db, account_id)
    if (
        account is None
        or account.status != "pending_deletion"
        or account.token_version != token_version
        or account.deletion_due_at is None
    ):
        raise AccountDeletionUnavailable()

    recovered_at = now or _utc_now_naive()
    if recovered_at >= account.deletion_due_at:
        raise AccountDeletionExpired()

    student = await _get_default_student(db, account.id)
    if student is None:
        raise AccountDeletionUnavailable()

    account.status = "active"
    account.deletion_requested_at = None
    account.deletion_due_at = None
    account.token_version += 1
    return account, student
