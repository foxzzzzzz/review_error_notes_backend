"""pytest fixtures — tests run against a live backend (docker-compose up first)."""
import pytest
import httpx
import uuid

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session")
def client():
    """httpx client against the running FastAPI backend."""
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


@pytest.fixture(scope="session")
def async_client():
    """async httpx client."""
    async def _client():
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as c:
            yield c
    return _client


@pytest.fixture
def test_user(client):
    """Create a unique test user via dev-login, return (token, student_id)."""
    openid = f"test_{uuid.uuid4().hex[:12]}"
    resp = client.post("/api/auth/dev-login", json={"code": openid})
    assert resp.status_code == 200, f"Dev login failed: {resp.text}"
    data = resp.json()
    return {"token": data["token"], "student_id": data["student_id"]}


@pytest.fixture
def auth_header(test_user):
    """Authorization header for the test user."""
    return {"Authorization": f"Bearer {test_user['token']}"}
