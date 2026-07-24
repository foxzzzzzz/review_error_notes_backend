from dataclasses import dataclass

import httpx


JSCODE2SESSION_URL = "https://api.weixin.qq.com/sns/jscode2session"
STABLE_TOKEN_URL = "https://api.weixin.qq.com/cgi-bin/stable_token"
PHONE_NUMBER_URL = "https://api.weixin.qq.com/wxa/business/getuserphonenumber"


class WeChatAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeChatSession:
    openid: str
    unionid: str | None = None


async def exchange_login_code(
    code: str,
    app_id: str,
    app_secret: str,
    client: httpx.AsyncClient = None,
) -> WeChatSession:
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=15)

    try:
        response = await client.get(
            JSCODE2SESSION_URL,
            params={
                "appid": app_id,
                "secret": app_secret,
                "js_code": code,
                "grant_type": "authorization_code",
            },
        )
        data = _response_json(response, "Unable to complete WeChat login")
        if data.get("errcode") not in (None, 0) or not data.get("openid"):
            raise WeChatAPIError("Unable to complete WeChat login")
        return WeChatSession(
            openid=data["openid"],
            unionid=data.get("unionid"),
        )
    except httpx.HTTPError as exc:
        raise WeChatAPIError("Unable to contact WeChat") from exc
    finally:
        if owns_client:
            await client.aclose()


async def get_phone_number(
    code: str,
    app_id: str,
    app_secret: str,
    client: httpx.AsyncClient = None,
) -> str:
    """Exchange a one-time WeChat code for a verified phone number."""
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=15)

    try:
        token_response = await client.post(
            STABLE_TOKEN_URL,
            json={
                "grant_type": "client_credential",
                "appid": app_id,
                "secret": app_secret,
            },
        )
        token_data = _response_json(token_response, "Unable to get WeChat access token")
        access_token = token_data.get("access_token")
        if not access_token:
            raise WeChatAPIError("Unable to get WeChat access token")

        phone_response = await client.post(
            PHONE_NUMBER_URL,
            params={"access_token": access_token},
            json={"code": code},
        )
        phone_data = _response_json(phone_response, "Unable to verify WeChat phone number")
        phone_info = phone_data.get("phone_info") or {}
        phone_number = phone_info.get("phoneNumber")
        if phone_data.get("errcode") not in (None, 0) or not phone_number:
            raise WeChatAPIError("Unable to verify WeChat phone number")
        return phone_number
    except httpx.HTTPError as exc:
        raise WeChatAPIError("Unable to contact WeChat") from exc
    finally:
        if owns_client:
            await client.aclose()


def _response_json(response: httpx.Response, error_message: str) -> dict:
    if response.status_code < 200 or response.status_code >= 300:
        raise WeChatAPIError(error_message)
    try:
        data = response.json()
    except ValueError as exc:
        raise WeChatAPIError(error_message) from exc
    if not isinstance(data, dict):
        raise WeChatAPIError(error_message)
    return data
