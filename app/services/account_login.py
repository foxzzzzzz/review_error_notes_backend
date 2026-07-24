from datetime import datetime, timezone

from sqlalchemy import select, text

from app.models.account import Account
from app.models.student import Student
from app.models.wechat_identity import WeChatIdentity


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def login_with_identity(
    db,
    appid: str,
    openid: str,
    unionid: str | None = None,
) -> tuple[Account, Student, bool]:
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:identity_key, 0))"),
        {"identity_key": f"{len(appid)}:{appid}{openid}"},
    )
    result = await db.execute(
        select(WeChatIdentity).where(
            WeChatIdentity.appid == appid,
            WeChatIdentity.openid == openid,
        )
    )
    identity = result.scalar_one_or_none()
    now = _utc_now_naive()

    if identity is not None:
        identity.last_login_at = now
        account_result = await db.execute(
            select(Account)
            .where(Account.id == identity.account_id)
            .with_for_update()
        )
        account = account_result.scalar_one_or_none()
        student_result = await db.execute(
            select(Student).where(
                Student.account_id == identity.account_id,
                Student.is_default.is_(True),
            )
        )
        student = student_result.scalar_one_or_none()
        if account is None or student is None:
            raise RuntimeError("Account identity is incomplete")
        return account, student, False

    account = Account()
    db.add_all([account])
    await db.flush()
    identity = WeChatIdentity(
        account_id=account.id,
        appid=appid,
        openid=openid,
        unionid=unionid,
        last_login_at=now,
    )
    student = Student(
        account_id=account.id,
        is_default=True,
        profile_completed=False,
    )
    db.add_all([identity, student])
    await db.flush()
    return account, student, True
