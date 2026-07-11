from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://wb_user:wb_pass_2024@localhost:5432/wrong_book"
    REDIS_URL: str = "redis://localhost:6379/0"
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = 7
    AES_KEY: str = "change-me-32bytes-secret-key-ok!"
    LLM_API_KEY: str = ""
    LLM_API_BASE: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    UPLOAD_DIR: str = "./uploads"
    PDF_DIR: str = "./pdfs"
    WECHAT_APP_ID: str = ""
    WECHAT_APP_SECRET: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
