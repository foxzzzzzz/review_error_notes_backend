from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from jose import JWTError, jwt
from sqlalchemy import exists, or_, select, text

from app.config import settings
from app.models.account import Account
from app.models.account_recovery import (
    AccountRecoveryConflict,
    FileCleanupJob,
)
from app.models.practice_attempt import PracticeAttempt
from app.models.practice_sheet import PracticeSheet
from app.models.student import Student
from app.models.wechat_identity import WeChatIdentity
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion
from app.utils.crypto import (
    encrypt_phone,
    fingerprint_phone,
    mask_phone,
    normalize_phone,
)


ACCOUNT_RECOVERY_SCOPE = "account_link_recovery"


@dataclass(frozen=True)
class BindPhoneResult:
    status: str
    phone_masked: str


@dataclass(frozen=True)
class AccountRecoveryClaims:
    current_account_id: str
    target_account_id: str
    phone_fingerprint: str
    current_token_version: int
    target_token_version: int


@dataclass(frozen=True)
class AccountRecoveryResult:
    account: Account
    student: Student
    cleanup_job: FileCleanupJob | None


class AccountRecoveryAvailable(RuntimeError):
    def __init__(self, recovery_token: str, phone_masked: str):
        super().__init__("Account recovery is available")
        self.recovery_token = recovery_token
        self.phone_masked = phone_masked


class AccountMergeRequired(RuntimeError):
    def __init__(self, support_reference: str):
        super().__init__("Manual account merge is required")
        self.support_reference = support_reference


class InvalidAccountRecovery(RuntimeError):
    pass


class TargetAccountPendingDeletion(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_naive() -> datetime:
    return _utc_now().replace(tzinfo=None)


def create_account_recovery_token(
    *,
    current_account,
    target_account,
    phone_fingerprint: str,
) -> str:
    expires_at = _utc_now() + timedelta(
        minutes=settings.ACCOUNT_RECOVERY_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(
        {
            "scope": ACCOUNT_RECOVERY_SCOPE,
            "current_account_id": str(current_account.id),
            "target_account_id": str(target_account.id),
            "phone_fingerprint": phone_fingerprint,
            "current_token_version": current_account.token_version,
            "target_token_version": target_account.token_version,
            "exp": expires_at,
        },
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )


def verify_account_recovery_token(
    token: str,
) -> AccountRecoveryClaims | None:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("scope") != ACCOUNT_RECOVERY_SCOPE:
            return None
        current_account_id = payload.get("current_account_id")
        target_account_id = payload.get("target_account_id")
        phone_fingerprint = payload.get("phone_fingerprint")
        current_token_version = payload.get("current_token_version")
        target_token_version = payload.get("target_token_version")
        if (
            not current_account_id
            or not target_account_id
            or not phone_fingerprint
            or not isinstance(current_token_version, int)
            or not isinstance(target_token_version, int)
        ):
            return None
        return AccountRecoveryClaims(
            current_account_id=str(current_account_id),
            target_account_id=str(target_account_id),
            phone_fingerprint=str(phone_fingerprint),
            current_token_version=current_token_version,
            target_token_version=target_token_version,
        )
    except JWTError:
        return None


async def account_has_business_data(db, account_id) -> bool:
    student_ids = select(Student.id).where(Student.account_id == account_id)
    statement = select(
        or_(
            exists(
                select(1).where(WrongImage.student_id.in_(student_ids))
            ),
            exists(
                select(1).where(WrongQuestion.student_id.in_(student_ids))
            ),
            exists(
                select(1).where(PracticeSheet.student_id.in_(student_ids))
            ),
            exists(
                select(1).where(PracticeAttempt.student_id.in_(student_ids))
            ),
            exists(
                select(1).where(
                    Student.account_id == account_id,
                    Student.is_default.is_(False),
                )
            ),
        )
    )
    return bool(await db.scalar(statement))


async def _lock_phone_fingerprint(db, phone_fingerprint: str) -> None:
    await db.execute(
        text(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended(:phone_fingerprint, 0))"
        ),
        {"phone_fingerprint": phone_fingerprint},
    )


async def _find_account_id_by_phone(db, phone_fingerprint: str):
    return await db.scalar(
        select(Account.id).where(
            Account.phone_fingerprint == phone_fingerprint
        )
    )


async def _lock_accounts(db, account_ids) -> dict[str, Account]:
    accounts = {}
    for account_id in sorted({str(value) for value in account_ids}):
        account = await db.scalar(
            select(Account)
            .where(Account.id == account_id)
            .with_for_update()
        )
        if account is not None:
            accounts[str(account.id)] = account
    return accounts


async def _lock_identities(db, account_id) -> list[WeChatIdentity]:
    result = await db.scalars(
        select(WeChatIdentity)
        .where(WeChatIdentity.account_id == account_id)
        .order_by(WeChatIdentity.id)
        .with_for_update()
    )
    return list(result.all())


async def _lock_students(db, account_id) -> list[Student]:
    result = await db.scalars(
        select(Student)
        .where(Student.account_id == account_id)
        .order_by(Student.id)
        .with_for_update()
    )
    return list(result.all())


async def bind_phone_to_account(
    db,
    current_account_id,
    phone: str,
) -> BindPhoneResult:
    normalized = normalize_phone(phone)
    phone_fingerprint = fingerprint_phone(normalized)
    await _lock_phone_fingerprint(db, phone_fingerprint)
    target_account_id = await _find_account_id_by_phone(
        db,
        phone_fingerprint,
    )
    account_ids = [current_account_id]
    if target_account_id is not None:
        account_ids.append(target_account_id)
    accounts = await _lock_accounts(db, account_ids)
    current_account = accounts.get(str(current_account_id))
    if current_account is None or current_account.status != "active":
        raise InvalidAccountRecovery("Current account is unavailable")

    if current_account.phone_fingerprint == phone_fingerprint:
        return BindPhoneResult(
            status="bound",
            phone_masked=mask_phone(normalized),
        )

    if target_account_id is None:
        current_account.phone_ciphertext = encrypt_phone(normalized)
        current_account.phone_fingerprint = phone_fingerprint
        current_account.phone_bound_at = _utc_now_naive()
        return BindPhoneResult(
            status="bound",
            phone_masked=mask_phone(normalized),
        )

    target_account = accounts.get(str(target_account_id))
    if (
        target_account is None
        or target_account.phone_fingerprint != phone_fingerprint
    ):
        raise InvalidAccountRecovery("Phone binding changed")

    if target_account.status == "pending_deletion":
        raise TargetAccountPendingDeletion()
    if target_account.status != "active":
        raise InvalidAccountRecovery("Target account is unavailable")

    if await account_has_business_data(db, current_account.id):
        support_reference = f"AR-{uuid4().hex[:12].upper()}"
        db.add(
            AccountRecoveryConflict(
                support_reference=support_reference,
                current_account_id=current_account.id,
                target_account_id=target_account.id,
                status="open",
            )
        )
        await db.flush()
        raise AccountMergeRequired(
            support_reference=support_reference
        )

    raise AccountRecoveryAvailable(
        recovery_token=create_account_recovery_token(
            current_account=current_account,
            target_account=target_account,
            phone_fingerprint=phone_fingerprint,
        ),
        phone_masked=mask_phone(normalized),
    )


async def recover_empty_account(
    db,
    recovery_token: str,
) -> AccountRecoveryResult:
    claims = verify_account_recovery_token(recovery_token)
    if claims is None:
        raise InvalidAccountRecovery("Invalid recovery token")

    await _lock_phone_fingerprint(db, claims.phone_fingerprint)
    accounts = await _lock_accounts(
        db,
        [claims.current_account_id, claims.target_account_id],
    )
    current_account = accounts.get(claims.current_account_id)
    target_account = accounts.get(claims.target_account_id)
    if (
        current_account is None
        or target_account is None
        or current_account.id == target_account.id
        or current_account.status != "active"
        or target_account.status != "active"
        or current_account.token_version != claims.current_token_version
        or target_account.token_version != claims.target_token_version
        or target_account.phone_fingerprint != claims.phone_fingerprint
    ):
        raise InvalidAccountRecovery("Recovery state changed")
    identities = await _lock_identities(db, current_account.id)
    current_students = await _lock_students(db, current_account.id)
    target_students = await _lock_students(db, target_account.id)
    if await account_has_business_data(db, current_account.id):
        raise InvalidAccountRecovery("Current account is no longer empty")

    target_default_students = [
        student for student in target_students if student.is_default
    ]
    if (
        len(identities) != 1
        or len(current_students) != 1
        or not current_students[0].is_default
        or len(target_default_students) != 1
    ):
        raise InvalidAccountRecovery("Recovery account structure changed")

    identities[0].account_id = target_account.id
    current_account.token_version += 1
    target_account.token_version += 1
    cleanup_job = None
    avatar_object_key = getattr(current_account, "avatar_object_key", None)
    if avatar_object_key and avatar_object_key.startswith("/avatars/"):
        cleanup_job = FileCleanupJob(
            storage_kind="avatar",
            object_path=avatar_object_key,
            attempt_count=0,
        )
        db.add(cleanup_job)
    await db.flush()
    await db.delete(current_students[0])
    await db.delete(current_account)
    await db.flush()
    return AccountRecoveryResult(
        account=target_account,
        student=target_default_students[0],
        cleanup_job=cleanup_job,
    )
