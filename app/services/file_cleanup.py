from pathlib import Path, PurePosixPath

from sqlalchemy import select

from app.config import settings
from app.models.account_recovery import FileCleanupJob


_STORAGE_LOCATIONS = {
    "avatar": ("/avatars", "AVATAR_DIR"),
    "upload": ("/uploads", "UPLOAD_DIR"),
    "pdf": ("/pdfs", "PDF_DIR"),
}


def _delete_cleanup_path(job: FileCleanupJob) -> None:
    location = _STORAGE_LOCATIONS.get(job.storage_kind)
    if location is None:
        raise ValueError("Unsupported cleanup storage kind")

    public_root, setting_name = location
    object_path = PurePosixPath(job.object_path)
    try:
        relative_path = object_path.relative_to(public_root)
    except ValueError as exc:
        raise ValueError("Cleanup path is outside its storage root") from exc
    if len(relative_path.parts) != 1 or relative_path.name in {"", ".", ".."}:
        raise ValueError("Cleanup path must identify one stored file")

    storage_root = Path(getattr(settings, setting_name)).resolve()
    target = (storage_root / relative_path.name).resolve()
    if target.parent != storage_root:
        raise ValueError("Cleanup path escaped its storage root")
    target.unlink(missing_ok=True)


async def attempt_file_cleanup(db, job_id) -> bool:
    job = await db.scalar(
        select(FileCleanupJob)
        .where(FileCleanupJob.id == job_id)
        .with_for_update()
    )
    if job is None:
        return True

    try:
        _delete_cleanup_path(job)
    except (OSError, ValueError) as exc:
        job.attempt_count += 1
        job.last_error = type(exc).__name__
        await db.commit()
        return False

    await db.delete(job)
    await db.commit()
    return True
