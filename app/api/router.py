from fastapi import APIRouter
from app.api.auth import router as auth_router
from app.api.upload import router as upload_router
from app.api.questions import router as questions_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(upload_router)
api_router.include_router(questions_router)
