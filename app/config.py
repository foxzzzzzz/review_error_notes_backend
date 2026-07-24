from pydantic import Field
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
    MINIMAX_API_KEY: str = ""
    MINIMAX_API_HOST: str = ""
    MINIMAX_VISION_TIMEOUT_SECONDS: float = 60
    MINIMAX_VISION_MAX_RETRIES: int = 2
    MINIMAX_VISION_RETRY_DELAY_SECONDS: float = 1
    MINIMAX_CONFIDENCE_THRESHOLD: float = 0.85
    MINIMAX_MARK_CONFIDENCE_THRESHOLD: float = Field(default=0.85, ge=0, le=1)
    MINIMAX_LOCALIZATION_CONFIDENCE_THRESHOLD: float = Field(default=0.85, ge=0, le=1)
    MINIMAX_LOCALIZATION_MAX_AREA_RATIO: float = Field(default=0.35, gt=0, le=1)
    QUESTION_CROP_CONTEXT_PADDING_RATIO: float = Field(default=0.15, ge=0, le=1)
    MARK_RED_PIXEL_MIN_RATIO: float = Field(default=0.005, ge=0, le=1)
    MARK_RED_PIXEL_EXPANSION_RATIO: float = Field(default=0.08, ge=0, le=1)
    MINIMAX_IMAGE_MAX_EDGE: int = 2048
    MINIMAX_IMAGE_JPEG_QUALITY: int = 90
    LOCAL_OCR_ENABLED: bool = True
    LOCAL_OCR_ENGINE: str = "onnxruntime"
    LOCAL_OCR_VERSION: str = "3.9.1"
    LOCAL_OCR_MODEL_VERSION: str = "PP-OCRv5"
    LOCAL_OCR_MODEL_TYPE: str = "mobile"
    LOCAL_OCR_MODEL_PATH: str = "./models/rapidocr"
    LOCAL_OCR_LINE_CONFIDENCE_THRESHOLD: float = Field(default=0.85, ge=0, le=1)
    LOCAL_OCR_MIN_EFFECTIVE_CHARACTERS: int = Field(default=2, ge=1)
    LOCAL_OCR_SUPPORT_SIMILARITY_THRESHOLD: float = Field(default=0.8, ge=0, le=1)
    LOCAL_OCR_CONTRADICTION_SIMILARITY_THRESHOLD: float = Field(
        default=0.9,
        ge=0,
        le=1,
    )
    TAG_ALIAS_CONFIG_PATH: str = "./config/tag-aliases.json"
    DEBUG_DATA_RESET_CONFIRMATION_PHRASE: str = "CLEAR_DEBUG_BUSINESS_DATA"
    QUESTION_IMAGE_MAX_PIXELS: int = 40_000_000
    QUESTION_SOFT_DELETE_RETENTION_DAYS: int = Field(
        default=30,
        ge=0,
        description="Days to retain soft-deleted questions and unreferenced images.",
    )
    QUESTION_CLEANUP_INTERVAL_SECONDS: int = Field(
        default=86_400,
        gt=0,
        description="Seconds between periodic cleanup runs.",
    )
    QUESTION_CLEANUP_BATCH_SIZE: int = Field(
        default=100,
        gt=0,
        description="Maximum records claimed by one cleanup query.",
    )
    UPLOAD_DIR: str = "./uploads"
    PDF_DIR: str = "./pdfs"
    WECHAT_APP_ID: str = ""
    WECHAT_APP_SECRET: str = ""
    DEV_MODE: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
