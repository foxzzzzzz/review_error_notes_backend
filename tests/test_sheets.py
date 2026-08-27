"""Tests for sheet generation and history."""
import pytest


class TestListSheets:
    """GET /api/sheets — empty list for new user."""

    def test_list_empty(self, client, auth_header):
        resp = client.get("/api/sheets", headers=auth_header)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestCreateSheet:
    """POST /api/sheets — requires existing questions."""

    def test_create_sheet_no_questions(self, client, auth_header):
        """Empty question_ids should return 400."""
        resp = client.post(
            "/api/sheets",
            json={
                "title": "测试卷",
                "question_ids": [],
                "derived_per_original": 1,
                "difficulty_boost": 2,
            },
            headers=auth_header,
        )
        assert resp.status_code == 400

    def test_create_sheet_invalid_question(self, client, auth_header):
        """Non-existent question_id should also return 400."""
        resp = client.post(
            "/api/sheets",
            json={
                "title": "测试卷",
                "question_ids": ["00000000-0000-0000-0000-000000000000"],
                "derived_per_original": 1,
                "difficulty_boost": 2,
            },
            headers=auth_header,
        )
        assert resp.status_code == 400

    def test_create_sheet_requires_auth(self, client):
        resp = client.post("/api/sheets", json={"title": "x", "question_ids": [], "derived_per_original": 1, "difficulty_boost": 1})
        assert resp.status_code == 401
