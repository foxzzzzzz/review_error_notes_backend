from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.schemas.auth import BindPhoneRequest, LoginRequest, LoginResponse
from app.services.account_login import login_with_identity
from app.services.wechat import WeChatAPIError, exchange_login_code
from app.utils.jwt import create_token


router = APIRouter(prefix="/auth", tags=["auth"])


def _login_response(account, student, is_new_account: bool) -> LoginResponse:
    return LoginResponse(
        token=create_token(str(account.id), account.token_version),
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


@router.post("/bind-phone")
async def bind_phone(_req: BindPhoneRequest):
    raise HTTPException(
        status_code=501,
        detail={
            "code": "phone_binding_unavailable",
            "message": "手机号绑定将在账户安全阶段开放",
        },
    )
