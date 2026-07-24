import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.router import api_router
from app.config import settings

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PDF_DIR, exist_ok=True)

app = FastAPI(title="Wrong Book API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/pdfs", StaticFiles(directory=settings.PDF_DIR), name="pdfs")


@app.get("/health")
def health():
    return {"status": "ok"}
