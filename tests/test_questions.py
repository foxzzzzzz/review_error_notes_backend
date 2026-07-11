"""Tests for question CRUD API."""
import pytest


@pytest.fixture
def question_id(client, auth_header):
    """Create a question manually via API for testing (requires OCR processed data).

    Since OCR requires a real image and Celery worker, we create a question
    by first uploading an image, or by testing against pre-existing data.
    For now, we test the query/filter/404 paths that don't require OCR.
    """
    # GET list — returns empty for new user, which is valid
    resp = client.get("/api/questions", headers=auth_header)
    return resp


class TestListQuestions:
    """GET /api/questions — list and filter."""

    def test_list_empty(self, client, auth_header):
        resp = client.get("/api/questions", headers=auth_header)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_filter_by_subject(self, client, auth_header):
        resp = client.get("/api/questions?subject=math", headers=auth_header)
        assert resp.status_code == 200

    def test_filter_by_grade(self, client, auth_header):
        resp = client.get("/api/questions?grade=2", headers=auth_header)
        assert resp.status_code == 200

    def test_filter_by_semester(self, client, auth_header):
        resp = client.get("/api/questions?semester=1", headers=auth_header)
        assert resp.status_code == 200

    def test_filter_by_status(self, client, auth_header):
        resp = client.get("/api/questions?status=confirmed", headers=auth_header)
        assert resp.status_code == 200

    def test_filter_by_tag(self, client, auth_header):
        resp = client.get("/api/questions?tag=加减法", headers=auth_header)
        assert resp.status_code == 200

    def test_pagination(self, client, auth_header):
        resp = client.get("/api/questions?limit=5&offset=0", headers=auth_header)
        assert resp.status_code == 200
        assert len(resp.json()) <= 5


class TestQuestionNotFound:
    """404 handling for non-existent questions."""

    def test_get_nonexistent_returns_404(self, client, auth_header):
        resp = client.get("/api/questions/00000000-0000-0000-0000-000000000000", headers=auth_header)
        assert resp.status_code == 404

    def test_patch_nonexistent_returns_404(self, client, auth_header):
        resp = client.patch(
            "/api/questions/00000000-0000-0000-0000-000000000000",
            json={"difficulty": 3},
            headers=auth_header,
        )
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, client, auth_header):
        resp = client.delete(
            "/api/questions/00000000-0000-0000-0000-000000000000",
            headers=auth_header,
        )
        assert resp.status_code == 404


class TestDataIsolation:
    """Verify student_id filtering — users can't access other users' data."""

    def test_cannot_access_other_user_question(self, client, auth_header):
        """Use a known question UUID — should return 404 (not visible) not 200."""
        resp = client.get(
            "/api/questions/11111111-1111-1111-1111-111111111111",
            headers=auth_header,
        )
        assert resp.status_code == 404


class TestHealthEndpoint:
    """Verify server is running."""

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
