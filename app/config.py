from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://wb_user:wb_pass_2024@localhost:5432/wrong_book"
    REDIS_URL: str = "redis://localhost:6379/0"
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = 7
    AES_KEY: str = "change-me-32bytes-secret-key-ok!"
    PHONE_HMAC_SECRET: str = "change-me-phone-hmac-secret"
    ACCOUNT_RECOVERY_TOKEN_EXPIRE_MINUTES: int = Field(default=10, gt=0)
    ACCOUNT_DELETION_RETENTION_DAYS: int = Field(default=30, ge=1)
    ACCOUNT_CLEANUP_INTERVAL_SECONDS: int = Field(
        default=86_400,
        gt=0,
        description="Seconds between expired account cleanup runs.",
    )
    ACCOUNT_CLEANUP_BATCH_SIZE: int = Field(
        default=50,
        gt=0,
        description="Maximum accounts or file jobs claimed per cleanup query.",
    )
    LLM_API_KEY: str = ""
    LLM_API_BASE: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_REQUEST_TIMEOUT_SECONDS: float = Field(
        default=60,
        gt=0,
        description="Timeout in seconds for one LLM HTTP request.",
    )
    MINIMAX_API_KEY: str = ""
    MINIMAX_API_HOST: str = ""
    MINIMAX_VISION_TIMEOUT_SECONDS: float = 60
    MINIMAX_VISION_MAX_RETRIES: int = 2
    MINIMAX_VISION_RETRY_DELAY_SECONDS: float = 1
    MINIMAX_THREE_STAGE_RECOGNITION_ENABLED: bool = True
    MINIMAX_MARK_STAGE_RETRY_COUNT: int = Field(default=1, ge=0, le=2)
    MINIMAX_LOCALIZATION_STAGE_RETRY_COUNT: int = Field(default=1, ge=0, le=2)
    MINIMAX_CONTENT_STAGE_RETRY_COUNT: int = Field(default=1, ge=0, le=2)
    MINIMAX_CONTENT_BATCH_SIZE: int = Field(default=6, ge=1, le=20)
    MINIMAX_CONFIDENCE_THRESHOLD: float = 0.85
    MINIMAX_MARK_CONFIDENCE_THRESHOLD: float = Field(default=0.85, ge=0, le=1)
    MINIMAX_LOCALIZATION_CONFIDENCE_THRESHOLD: float = Field(default=0.85, ge=0, le=1)
    MINIMAX_LOCALIZATION_MAX_AREA_RATIO: float = Field(default=0.35, gt=0, le=1)
    QUESTION_CROP_CONTEXT_PADDING_RATIO: float = Field(default=0.15, ge=0, le=1)
    MARK_RED_PIXEL_MIN_RATIO: float = Field(default=0.005, ge=0, le=1)
    MARK_RED_PIXEL_EXPANSION_RATIO: float = Field(default=0.08, ge=0, le=1)
    MARK_CORRECTION_GROUP_ENABLED: bool = True
    MARK_PAIR_MAX_DISTANCE_RATIO: float = Field(default=0.04, ge=0, le=1)
    MARK_ANCHOR_MAX_GAP_RATIO: float = Field(default=0.08, ge=0, le=1)
    MARK_CROSS_ONLY_MAX_GAP_RATIO: float = Field(default=0.08, ge=0, le=1)
    MARK_DEDUP_IOU_THRESHOLD: float = Field(default=0.8, gt=0, le=1)
    MINIMAX_IMAGE_MAX_EDGE: int = 2048
    MINIMAX_IMAGE_JPEG_QUALITY: int = 90
    MINIMAX_MARK_MISMATCH_RETRY_COUNT: int = Field(default=1, ge=0, le=2)
    MINIMAX_LOCALIZATION_SEMANTIC_RETRY_COUNT: int = Field(default=1, ge=0, le=2)
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
    LOCAL_OCR_FULL_PAGE_MAX_EDGE: int = Field(default=1600, ge=640, le=4096)
    LOCAL_OCR_CROP_RECHECK_LIMIT: int = Field(default=3, ge=0, le=20)
    LOCAL_OCR_MARKED_RECHECK_LIMIT: int = Field(default=3, ge=0, le=20)
    LOCAL_RED_SCAN_MAX_EDGE: int = Field(default=1600, ge=640, le=4096)
    LOCAL_RED_COMPONENT_MIN_PIXELS: int = Field(default=12, ge=1)
    LOCAL_RED_COMPONENT_MAX_AREA_RATIO: float = Field(default=0.08, gt=0, le=1)
    LOCAL_RED_COMPONENT_MAX_THINNESS_RATIO: float = Field(default=18, ge=1)
    LOCAL_RED_RESCUE_MIN_PIXELS: int = Field(default=80, ge=1)
    CELERY_WORKER_CONCURRENCY: int = Field(default=2, ge=1, le=16)
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
    SHEET_GENERATION_SOFT_TIME_LIMIT_SECONDS: int = Field(
        default=7200,
        gt=0,
        description="Soft time limit for one asynchronous sheet generation task.",
    )
    SHEET_DERIVATIVE_GENERATION_MODE: Literal["serial", "batch"] = "serial"
    SHEET_DERIVATIVE_BATCH_SIZE: int = Field(default=8, ge=1, le=20)
    SHEET_DERIVATIVE_MAX_CONCURRENCY: int = Field(default=3, ge=1, le=8)
    SHEET_DERIVATIVE_RESPONSE_VALIDATION_RETRY_COUNT: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Extra retries for malformed or schema-invalid derivative batch responses.",
    )
    UPLOAD_DIR: str = "./uploads"
    UPLOAD_MAX_BYTES: int = Field(default=10_485_760, gt=0)
    INCOMPLETE_IMAGE_STATUS_LIMIT: int = Field(default=100, ge=1, le=500)
    PDF_DIR: str = "./pdfs"
    AVATAR_DIR: str = "./avatars"
    AVATAR_MAX_BYTES: int = Field(default=5_242_880, gt=0)
    AVATAR_MAX_EDGE: int = Field(default=512, gt=0)
    AVATAR_JPEG_QUALITY: int = Field(default=88, ge=1, le=100)
    WECHAT_APP_ID: str = ""
    WECHAT_APP_SECRET: str = ""
    DEV_MODE: bool = False
    APP_ENV: str = "development"
    DEV_LOGIN_IDENTITY: str = "dev-local-account"
    DEV_LOGIN_ALLOWED_ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()
