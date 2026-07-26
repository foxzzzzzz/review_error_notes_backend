from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AccountContext, get_current_account
from app.config import settings
from app.database import get_db
from app.schemas.auth import (
    BindPhoneRequest,
    BindPhoneResponse,
    LoginRequest,
    LoginResponse,
    RecoverAccountRequest,
)
from app.services.account_login import login_with_identity
from app.services.account_recovery import (
    AccountMergeRequired,
    AccountRecoveryAvailable,
    InvalidAccountRecovery,
    TargetAccountPendingDeletion,
    bind_phone_to_account,
    recover_empty_account,
)
from app.services.file_cleanup import attempt_file_cleanup
from app.services.wechat import (
    WeChatAPIError,
    exchange_login_code,
    get_phone_number,
)
from app.utils.jwt import create_account_recovery_token, create_token


router = APIRouter(prefix="/auth", tags=["auth"])


def _login_response(account, student, is_new_account: bool) -> LoginResponse:
    if account.status == "pending_deletion":
        recovery_token = None
        if account.deletion_due_at is not None:
            deletion_due_at = account.deletion_due_at
            if deletion_due_at.tzinfo is None:
                deletion_due_at = deletion_due_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) >= deletion_due_at:
                raise HTTPException(
                    status_code=410,
                    detail={
                        "code": "account_deletion_expired",
                        "message": "账户恢复期限已过",
                    },
                )
            recovery_token = create_account_recovery_token(
                str(account.id),
                account.token_version,
                deletion_due_at,
            )
        return LoginResponse(
            token=None,
            recovery_token=recovery_token,
            account_id=account.id,
            student_id=student.id,
            is_new_account=False,
            profile_prompt_required=False,
            student_profile_required=False,
            account_status=account.status,
            deletion_due_at=deletion_due_at,
        )
    return LoginResponse(
        token=create_token(str(account.id), account.token_version),
        recovery_token=None,
        account_id=account.id,
        student_id=student.id,
        is_new_account=is_new_account,
        profile_prompt_required=(
            account.profile_prompted_at is None
            and account.profile_completed_at is None
        ),
        student_profile_required=(
            student.grade is None or student.semester is None
        ),
        account_status=account.status,
    )


async def _complete_login(
    db: AsyncSession,
    appid: str,
    openid: str,
    unionid: str | None = None,
) -> LoginResponse:
    try:
        account, student, created = await login_with_identity(
            db,
            appid=appid,
            openid=openid,
            unionid=unionid,
        )
        response = _login_response(account, student, created)
        await db.commit()
        return response
    except Exception:
        await db.rollback()
        raise


@router.post("/login", response_model=LoginResponse)
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        session = await exchange_login_code(
            req.code,
            settings.WECHAT_APP_ID,
            settings.WECHAT_APP_SECRET,
        )
    except WeChatAPIError as exc:
        raise HTTPException(
            status_code=502,
            detail="Unable to complete WeChat login",
        ) from exc
    return await _complete_login(
        db,
        appid=settings.WECHAT_APP_ID,
        openid=session.openid,
        unionid=session.unionid,
    )


@router.post("/dev-login", response_model=LoginResponse)
async def dev_login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    if (
        not settings.DEV_MODE
        or settings.APP_ENV != settings.DEV_LOGIN_ALLOWED_ENVIRONMENT
    ):
        raise HTTPException(
            status_code=403,
            detail="Dev login is unavailable",
        )
    if not req.code:
        raise HTTPException(status_code=400, detail="code is required")
    return await _complete_login(
        db,
        appid=f"dev:{settings.WECHAT_APP_ID or 'local'}",
        openid=req.code,
    )


@router.post("/bind-phone", response_model=BindPhoneResponse)
async def bind_phone(
    req: BindPhoneRequest,
    context: AccountContext = Depends(get_current_account),
    db: AsyncSession = Depends(get_db),
):
    try:
        phone = await get_phone_number(
            req.code,
            settings.WECHAT_APP_ID,
            settings.WECHAT_APP_SECRET,
        )
        result = await bind_phone_to_account(
            db,
            context.account_id,
            phone,
        )
        await db.commit()
        return BindPhoneResponse(
            status="bound",
            phone_masked=result.phone_masked,
        )
    except WeChatAPIError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=502,
            detail={
                "code": "phone_verification_failed",
                "message": "手机号验证失败，请重试",
            },
        ) from exc
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=502,
            detail={
                "code": "phone_verification_failed",
                "message": "手机号格式无效，请重试",
            },
        ) from exc
    except AccountRecoveryAvailable as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "account_recovery_available",
                "message": "该手机号关联了已有账户，是否恢复原账户？",
                "phone_masked": exc.phone_masked,
                "recovery_token": exc.recovery_token,
            },
        ) from exc
    except AccountMergeRequired as exc:
        await db.commit()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "account_merge_required",
                "message": "两个账户均有数据，请联系人工处理",
                "support_reference": exc.support_reference,
            },
        ) from exc
    except TargetAccountPendingDeletion as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "account_pending_deletion",
                "message": "原账户正在注销流程中，请先恢复原账户",
            },
        ) from exc
    except InvalidAccountRecovery as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "phone_binding_conflict",
                "message": "账户状态已变化，请重新操作",
            },
        ) from exc


@router.post("/recover-account", response_model=LoginResponse)
async def recover_account(
    req: RecoverAccountRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await recover_empty_account(
            db,
            req.recovery_token,
        )
        response = _login_response(
            result.account,
            result.student,
            False,
        )
        await db.commit()
        if result.cleanup_job is not None:
            try:
                await attempt_file_cleanup(db, result.cleanup_job.id)
            except Exception:
                await db.rollback()
        return response
    except InvalidAccountRecovery as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "account_recovery_invalid",
                "message": "恢复凭证已失效，请重新绑定手机号",
            },
        ) from exc
