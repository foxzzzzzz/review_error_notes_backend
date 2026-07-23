"""Clear debug business data while preserving student accounts."""

import argparse
from pathlib import Path

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from app.config import settings
from app.models.practice_sheet import PracticeSheet
from app.models.sheet_item import SheetItem
from app.models.wrong_image import WrongImage
from app.models.wrong_question import WrongQuestion


BUSINESS_MODELS_IN_DELETE_ORDER = (
    SheetItem,
    PracticeSheet,
    WrongQuestion,
    WrongImage,
)


def confirmation_matches(provided: str, expected: str) -> bool:
    return bool(provided) and provided == expected


def reset_business_records(session) -> None:
    for model in BUSINESS_MODELS_IN_DELETE_ORDER:
        session.execute(delete(model))
    session.commit()


def clear_storage_files(directory: str) -> int:
    root = Path(directory).resolve()
    if root.name not in {"uploads", "pdfs"}:
        raise ValueError("storage directory must resolve to uploads or pdfs")
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
        description="Clear wrong-question debug data but preserve students."
    )
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()

    expected = settings.DEBUG_DATA_RESET_CONFIRMATION_PHRASE
    if not confirmation_matches(args.confirm, expected):
        parser.error("confirmation phrase does not match configured value")

    sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(sync_url)
    try:
        with Session(engine) as session:
            reset_business_records(session)
        uploads_removed = clear_storage_files(settings.UPLOAD_DIR)
        pdfs_removed = clear_storage_files(settings.PDF_DIR)
    finally:
        engine.dispose()

    print(
        "Debug business data cleared; students preserved; "
        f"uploads removed={uploads_removed}; pdfs removed={pdfs_removed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
