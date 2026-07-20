"""pytest fixtures — tests run against a live backend (docker-compose up first)."""
import pytest
import httpx
import uuid
import time

BASE_URL = "http://localhost:8000"


@pytest.fixture
def client():
    """httpx client against the running FastAPI backend (fresh per test)."""
    # Wait for API to be ready
    for _ in range(5):
        try:
            with httpx.Client(base_url=BASE_URL, timeout=10) as c:
                c.get("/health")
            break
        except Exception:
            time.sleep(0.5)
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


@pytest.fixture
def test_user(client):
    """Create a unique test user via dev-login, return (token, student_id)."""
    openid = f"test_{uuid.uuid4().hex[:12]}"
    # Retry up to 3 times in case of transient connection issues
    for attempt in range(3):
        try:
            resp = client.post("/api/auth/dev-login", json={"code": openid})
            assert resp.status_code == 200, f"Dev login failed: {resp.text}"
            data = resp.json()
            return {"token": data["token"], "student_id": data["student_id"]}
        except httpx.ConnectError:
            if attempt == 2:
                raise
            time.sleep(0.3)


@pytest.fixture
def auth_header(test_user):
    """Authorization header for the test user."""
    return {"Authorization": f"Bearer {test_user['token']}"}
