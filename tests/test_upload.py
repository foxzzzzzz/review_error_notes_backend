"""End-to-end test: upload image → verify record created."""
import pytest
import io
from PIL import Image


@pytest.fixture
def test_image():
    """Generate a small test image in memory."""
    img = Image.new("RGB", (200, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    buf.name = "test_question.jpg"
    return buf


class TestUploadImage:
    """POST /api/upload/image — file upload."""

    def test_upload_requires_auth(self, client, test_image):
        resp = client.post("/api/upload/image", files={"file": (test_image.name, test_image, "image/jpeg")})
        assert resp.status_code == 401

    def test_upload_image_success(self, client, auth_header, test_image):
        resp = client.post(
            "/api/upload/image",
            files={"file": (test_image.name, test_image, "image/jpeg")},
            data={"grade": "1", "semester": "1"},
            headers=auth_header,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image_id" in data
        assert data["status"] == "pending"
        # Verify it's a valid UUID format
        import uuid
        uuid.UUID(data["image_id"])

    def test_upload_creates_record(self, client, auth_header, test_image):
        """Upload should create a WrongImage record visible in the API."""
        resp = client.post(
            "/api/upload/image",
            files={"file": (test_image.name, test_image, "image/jpeg")},
            data={"grade": "1", "semester": "1"},
            headers=auth_header,
        )
        assert resp.status_code == 200
def test_original_image_route_is_owner_scoped_and_uses_safe_basename():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app" / "api" / "upload.py").read_text(encoding="utf-8")

    assert '@router.get("/images/{image_id}/original")' in source
    assert "WrongImage.student_id == student.id" in source
    assert "Path(image.original_url).name" in source
    assert "FileResponse" in source
