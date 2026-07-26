from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import _login_response
from app.api.deps import (
    AccountContext,
    get_current_account,
    get_deletion_account,
    get_recovery_account,
)
from app.config import settings
from app.database import get_db
from app.schemas.account import (
    AccountDeletionRequest,
    AccountDeletionResponse,
    OperationResponse,
)
from app.schemas.auth import LoginResponse
from app.services.account_deletion import (
    AccountDeletionExpired,
    AccountDeletionUnavailable,
    InvalidDeletionIdentity,
    recover_account_deletion,
    request_account_deletion,
)
from app.services.wechat import WeChatAPIError, exchange_login_code
from app.utils.jwt import create_account_recovery_token


router = APIRouter(prefix="/account", tags=["account"])


async def _resolve_fresh_identity(code: str) -> tuple[str, str]:
    if (
        settings.DEV_MODE
        and settings.APP_ENV == settings.DEV_LOGIN_ALLOWED_ENVIRONMENT
    ):
        return (
            f"dev:{settings.WECHAT_APP_ID or 'local'}",
            code,
        )
    session = await exchange_login_code(
        code,
        settings.WECHAT_APP_ID,
        settings.WECHAT_APP_SECRET,
    )
    return settings.WECHAT_APP_ID, session.openid


@router.post("/logout", response_model=OperationResponse)
async def logout(
    context: AccountContext = Depends(get_current_account),
):
    return OperationResponse(ok=True)


@router.post(
    "/deletion",
    response_model=AccountDeletionResponse,
)
async def delete_account(
    req: AccountDeletionRequest,
    context: AccountContext = Depends(get_deletion_account),
    db: AsyncSession = Depends(get_db),
):
    try:
        appid, openid = await _resolve_fresh_identity(req.code)
        account = await request_account_deletion(
            db,
            context.account_id,
            appid=appid,
            openid=openid,
        )
        response = AccountDeletionResponse(
            account_status="pending_deletion",
            deletion_due_at=account.deletion_due_at,
            recovery_token=create_account_recovery_token(
                str(account.id),
                account.token_version,
                account.deletion_due_at,
            ),
        )
        await db.commit()
        return response
    except WeChatAPIError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=502,
            detail={
                "code": "wechat_login_failed",
                "message": "微信身份验证失败，请重试",
            },
        ) from exc
    except InvalidDeletionIdentity as exc:
        await db.rollback()
        raise HTTPException(
            status_code=403,
            detail={
                "code": "deletion_identity_mismatch",
                "message": "当前微信身份与账户不一致",
            },
        ) from exc
    except AccountDeletionUnavailable as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "account_deletion_unavailable",
                "message": "当前账户状态无法发起注销",
            },
        ) from exc


@router.post(
    "/deletion/recover",
    response_model=LoginResponse,
)
async def recover_deleted_account(
    context: AccountContext = Depends(get_recovery_account),
    db: AsyncSession = Depends(get_db),
):
    try:
        account, student = await recover_account_deletion(
            db,
            context.account_id,
            token_version=context.token_version,
        )
        response = _login_response(account, student, False)
        await db.commit()
        return response
    except AccountDeletionExpired as exc:
        await db.rollback()
        raise HTTPException(
            status_code=410,
            detail={
                "code": "account_deletion_expired",
                "message": "账户恢复期限已过",
            },
        ) from exc
    except AccountDeletionUnavailable as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={
                "code": "account_recovery_invalid",
                "message": "账户状态已变化，请重新登录",
            },
        ) from exc
