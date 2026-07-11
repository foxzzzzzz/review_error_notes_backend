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
    # 微信手机号解密在后端完成(实际需调用微信解密接口)
    # TODO: 接入微信手机号解密API, 此处暂存加密数据
    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one()
    student.phone = encrypt_phone(req.encrypted_data)  # 实际为解密后的手机号
    student.phone_bound = True
    await db.commit()
    return {"ok": True}
