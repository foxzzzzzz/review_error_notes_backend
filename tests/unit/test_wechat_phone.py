import asyncio

import httpx
import pytest
from pydantic import ValidationError


def test_bind_phone_request_requires_one_time_code():
    from app.schemas.auth import BindPhoneRequest

    assert BindPhoneRequest(code="phone-code").model_dump() == {"code": "phone-code"}
    with pytest.raises(ValidationError):
        BindPhoneRequest(code="")


def test_wechat_phone_flow_uses_stable_token_and_returns_verified_number():
    from app.services.wechat import get_phone_number

    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/cgi-bin/stable_token":
            assert request.method == "POST"
            assert request.content == b'{"grant_type":"client_credential","appid":"app-id","secret":"app-secret"}'
            return httpx.Response(200, json={"access_token": "access-token", "expires_in": 7200})
        assert request.url.path == "/wxa/business/getuserphonenumber"
        assert request.url.params["access_token"] == "access-token"
        assert request.content == b'{"code":"phone-code"}'
        return httpx.Response(200, json={
            "errcode": 0,
            "errmsg": "ok",
            "phone_info": {"phoneNumber": "+8613800138000"},
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await get_phone_number("phone-code", "app-id", "app-secret", client=client)

    assert asyncio.run(run()) == "+8613800138000"
    assert len(requests) == 2


@pytest.mark.parametrize(
    "responses",
    (
        [{"errcode": 40013, "errmsg": "invalid appid"}],
        [
            {"access_token": "access-token", "expires_in": 7200},
            {"errcode": 40029, "errmsg": "invalid code"},
        ],
        [
            {"access_token": "access-token", "expires_in": 7200},
            {"errcode": 0, "errmsg": "ok"},
        ],
    ),
)
def test_wechat_phone_flow_rejects_malformed_upstream_responses(responses):
    from app.services.wechat import WeChatAPIError, get_phone_number

    queue = list(responses)

    def handler(_request):
        return httpx.Response(200, json=queue.pop(0))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await get_phone_number("phone-code", "app-id", "app-secret", client=client)

    with pytest.raises(WeChatAPIError):
        asyncio.run(run())
