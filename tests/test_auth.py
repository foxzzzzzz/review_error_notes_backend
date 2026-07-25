"""Tests for auth endpoints: dev-login, normal login, phone binding."""
import pytest


class TestDevLogin:
    """Development mode login — core test infrastructure."""

    def test_dev_login_creates_user(self, client):
        resp = client.post("/api/auth/dev-login", json={"code": "test_openid_001"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["is_new_account"] is True
        assert data["profile_prompt_required"] is True
        assert data["student_profile_required"] is True
        assert data["account_status"] == "active"
        assert "account_id" in data
        assert "student_id" in data

    def test_dev_login_returns_existing_user(self, client):
        """Second login with same openid returns same student."""
        openid = "test_repeat_user"
        r1 = client.post("/api/auth/dev-login", json={"code": openid})
        assert r1.status_code == 200
        account_id_1 = r1.json()["account_id"]
        student_id_1 = r1.json()["student_id"]

        r2 = client.post("/api/auth/dev-login", json={"code": openid})
        assert r2.status_code == 200
        account_id_2 = r2.json()["account_id"]
        student_id_2 = r2.json()["student_id"]
        assert account_id_1 == account_id_2
        assert student_id_1 == student_id_2

    def test_dev_login_requires_code(self, client):
        resp = client.post("/api/auth/dev-login", json={"code": ""})
        assert resp.status_code == 400


class TestTokenAuth:
    """JWT token validation."""

    def test_valid_token_accesses_protected_route(self, client, auth_header):
        resp = client.get("/api/questions", headers=auth_header)
        assert resp.status_code == 200

    def test_no_token_returns_401(self, client):
        resp = client.get("/api/questions")
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, client):
        resp = client.get("/api/questions", headers={"Authorization": "Bearer garbage_token"})
        assert resp.status_code == 401


class TestBindPhone:
    """Phone binding uses a WeChat one-time phone authorization code."""

    def test_bind_phone_requires_a_non_empty_code(self, client, auth_header):
        resp = client.post(
            "/api/auth/bind-phone",
            json={"code": ""},
            headers=auth_header,
        )
        assert resp.status_code == 422
