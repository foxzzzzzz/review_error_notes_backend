from io import BytesIO
from types import SimpleNamespace
import asyncio

import pytest
from PIL import Image

from app.services.avatar_storage import (
    AvatarInvalid,
    AvatarTooLarge,
    SavedAvatar,
    save_avatar_image,
)


def _image_bytes(size=(80, 40), image_format="PNG"):
    output = BytesIO()
    Image.new("RGB", size, "red").save(output, format=image_format)
    return output.getvalue()


def test_save_avatar_rejects_oversized_input(tmp_path):
    with pytest.raises(AvatarTooLarge):
        save_avatar_image(
            b"1234",
            avatar_dir=tmp_path,
            max_bytes=3,
            max_edge=64,
            jpeg_quality=85,
        )


def test_save_avatar_rejects_invalid_image(tmp_path):
    with pytest.raises(AvatarInvalid):
        save_avatar_image(
            b"not-an-image",
            avatar_dir=tmp_path,
            max_bytes=1024,
            max_edge=64,
            jpeg_quality=85,
        )


def test_save_avatar_center_crops_and_writes_square_jpeg(tmp_path):
    result = save_avatar_image(
        _image_bytes(),
        avatar_dir=tmp_path,
        max_bytes=1024 * 1024,
        max_edge=32,
        jpeg_quality=85,
    )

    assert result.public_url == f"/avatars/{result.filename}"
    assert result.path.exists()
    assert result.path.suffix == ".jpg"

    with Image.open(result.path) as avatar:
        assert avatar.format == "JPEG"
        assert avatar.size == (32, 32)


def test_avatar_endpoint_removes_new_file_when_database_lookup_fails(
    tmp_path,
    monkeypatch,
):
    from app.api import profile as profile_api

    avatar_path = tmp_path / "new.jpg"
    avatar_path.write_bytes(b"new")
    monkeypatch.setattr(
        profile_api,
        "save_avatar_image",
        lambda *_args, **_kwargs: SavedAvatar(
            filename="new.jpg",
            path=avatar_path,
            public_url="/avatars/new.jpg",
        ),
    )

    class File:
        async def read(self, _size):
            return b"image"

    class DB:
        rollback_calls = 0

        async def scalar(self, _statement):
            raise RuntimeError("database unavailable")

        async def rollback(self):
            self.rollback_calls += 1

    db = DB()
    with pytest.raises(RuntimeError, match="database unavailable"):
        asyncio.run(
            profile_api.upload_avatar(
                file=File(),
                student=SimpleNamespace(account_id="account-id"),
                db=db,
            )
        )

    assert db.rollback_calls == 1
    assert not avatar_path.exists()
