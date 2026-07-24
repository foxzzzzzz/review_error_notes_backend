"""Delete all test account/business data and generated files."""

import argparse
from pathlib import Path

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from app.config import settings
from app.models.account import Account
from app.models.practice_sheet import PracticeSheet
from app.models.sheet_item import SheetItem
from app.models.student import Student
from app.models.wechat_identity import WeChatIdentity
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion


TEST_MODELS_IN_DELETE_ORDER = (
    SheetItem,
    PracticeSheet,
    WrongQuestion,
    WrongImage,
    Student,
    WeChatIdentity,
    Account,
)


def confirmation_matches(provided: str, expected: str) -> bool:
    return bool(provided) and provided == expected


def assert_reset_allowed(app_env: str, provided: str, expected: str) -> None:
    if app_env not in {"development", "test"}:
        raise RuntimeError("full reset is limited to a test environment")
    if not confirmation_matches(provided, expected):
        raise ValueError("confirmation phrase does not match")


def reset_all_test_records(session) -> None:
    for model in TEST_MODELS_IN_DELETE_ORDER:
        session.execute(delete(model))
    session.commit()


def clear_storage_files(directory: str) -> int:
    root = Path(directory).resolve()
    if root.name not in {"uploads", "pdfs", "avatars"}:
        raise ValueError(
            "storage directory must resolve to uploads, pdfs, or avatars"
        )
    if not root.is_dir():
        return 0

    removed = 0
    for child in root.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete all account and business data in a test environment."
    )
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()

    assert_reset_allowed(
        settings.APP_ENV,
        args.confirm,
        settings.DEBUG_DATA_RESET_CONFIRMATION_PHRASE,
    )

    sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(sync_url)
    try:
        with Session(engine) as session:
            reset_all_test_records(session)
        uploads_removed = clear_storage_files(settings.UPLOAD_DIR)
        pdfs_removed = clear_storage_files(settings.PDF_DIR)
        avatars_removed = clear_storage_files(settings.AVATAR_DIR)
    finally:
        engine.dispose()

    print(
        "All test account and business data cleared; "
        f"uploads removed={uploads_removed}; "
        f"pdfs removed={pdfs_removed}; "
        f"avatars removed={avatars_removed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
