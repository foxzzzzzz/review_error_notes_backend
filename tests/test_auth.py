"""Tests for auth endpoints: dev-login, normal login, phone binding."""
import pytest


class TestDevLogin:
    """Development mode login — core test infrastructure."""

    def test_dev_login_creates_user(self, client):
        resp = client.post("/api/auth/dev-login", json={"code": "test_openid_001"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["need_phone"] is True
        assert "student_id" in data

    def test_dev_login_returns_existing_user(self, client):
        """Second login with same openid returns same student."""
        openid = "test_repeat_user"
        r1 = client.post("/api/auth/dev-login", json={"code": openid})
        assert r1.status_code == 200
        id1 = r1.json()["student_id"]

        r2 = client.post("/api/auth/dev-login", json={"code": openid})
        assert r2.status_code == 200
        id2 = r2.json()["student_id"]
        assert id1 == id2

    def test_dev_login_requires_code(self, client):
        resp = client.post("/api/auth/dev-login", json={"code": ""})
        assert resp.status_code == 400


class TestTokenAuth:
    """JWT token validation."""

    def test_valid_token_accesses_protected_route(self, client, auth_header):
        resp = client.get("/api/questions", headers=auth_header)
        assert resp.status_code == 200

    def test_no_token_returns_403(self, client):
        resp = client.get("/api/questions")
        assert resp.status_code == 403

    def test_invalid_token_returns_401(self, client):
        resp = client.get("/api/questions", headers={"Authorization": "Bearer garbage_token"})
        assert resp.status_code == 401


class TestBindPhone:
    """Phone number binding."""

    def test_bind_phone(self, client, auth_header):
        resp = client.post(
            "/api/auth/bind-phone",
            json={"encrypted_data": "13800138000", "iv": "test_iv"},
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
