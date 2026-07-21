import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.student import Student
from app.schemas.auth import LoginRequest, LoginResponse, BindPhoneRequest
from app.utils.jwt import create_token
from app.utils.crypto import encrypt_phone
from app.api.deps import get_current_student
from app.config import settings
from app.services.wechat import WeChatAPIError, get_phone_number

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    # 用 code 换 openid
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.weixin.qq.com/sns/jscode2session",
            params={
                "appid": settings.WECHAT_APP_ID,
                "secret": settings.WECHAT_APP_SECRET,
                "js_code": req.code,
                "grant_type": "authorization_code",
            },
        )
    wx_data = resp.json()
    openid = wx_data.get("openid")
    if not openid:
        raise HTTPException(status_code=400, detail=f"WeChat login failed: {wx_data}")

    # 查或创学生
    result = await db.execute(select(Student).where(Student.openid == openid))
    student = result.scalar_one_or_none()
    if student:
        return LoginResponse(
            token=create_token(str(student.id)),
            need_phone=not student.phone_bound,
            student_id=str(student.id),
        )
    student = Student(openid=openid)
    db.add(student)
    await db.commit()
    await db.refresh(student)
    return LoginResponse(
        token=create_token(str(student.id)),
        need_phone=True,
        student_id=str(student.id),
    )


@router.post("/bind-phone")
async def bind_phone(req: BindPhoneRequest, student_id=Depends(get_current_student), db=Depends(get_db)):
    try:
        phone_number = await get_phone_number(
            req.code,
            settings.WECHAT_APP_ID,
            settings.WECHAT_APP_SECRET,
        )
    except WeChatAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one()
    student.phone = encrypt_phone(phone_number)
    student.phone_bound = True
    await db.commit()
    return {"ok": True}


@router.post("/dev-login", response_model=LoginResponse)
async def dev_login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """开发模式：绕过微信直接登录（仅在 DEV_MODE=true 时可用）"""
    if not settings.DEV_MODE:
        raise HTTPException(status_code=403, detail="Dev login only available in DEV_MODE")
    if not req.code:
        raise HTTPException(status_code=400, detail="code is required (used as openid in dev mode)")

    openid = req.code  # dev 模式下 code 即 openid
    result = await db.execute(select(Student).where(Student.openid == openid))
    student = result.scalar_one_or_none()
    if student:
        return LoginResponse(
            token=create_token(str(student.id)),
            need_phone=not student.phone_bound,
            student_id=str(student.id),
        )
    student = Student(openid=openid, nickname=openid)
    db.add(student)
    await db.commit()
    await db.refresh(student)
    return LoginResponse(
        token=create_token(str(student.id)),
        need_phone=True,
        student_id=str(student.id),
    )
