from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError


class AvatarTooLarge(ValueError):
    pass


class AvatarInvalid(ValueError):
    pass


@dataclass(frozen=True)
class SavedAvatar:
    filename: str
    path: Path
    public_url: str


def save_avatar_image(
    data: bytes,
    *,
    avatar_dir: str | Path,
    max_bytes: int,
    max_edge: int,
    jpeg_quality: int,
) -> SavedAvatar:
    if len(data) > max_bytes:
        raise AvatarTooLarge

    target_dir = Path(avatar_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4()}.jpg"
    target_path = target_dir / filename

    try:
        with Image.open(BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            avatar = ImageOps.fit(
                image,
                (max_edge, max_edge),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            avatar.save(
                target_path,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        target_path.unlink(missing_ok=True)
        raise AvatarInvalid from exc

    return SavedAvatar(
        filename=filename,
        path=target_path,
        public_url=f"/avatars/{filename}",
    )
