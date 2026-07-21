from fastapi import APIRouter
from app.api.auth import router as auth_router
from app.api.upload import router as upload_router
from app.api.questions import router as questions_router
from app.api.sheets import router as sheets_router
from app.api.profile import router as profile_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(upload_router)
api_router.include_router(questions_router)
api_router.include_router(sheets_router)
api_router.include_router(profile_router)
