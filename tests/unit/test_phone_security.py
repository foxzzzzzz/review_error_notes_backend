import base64
from pathlib import Path
import re

import pytest


BACKEND_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "raw_phone",
    (
        "13800138000",
        "+86 138 0013 8000",
        "86-138-0013-8000",
        "138-0013-8000",
    ),
)
def test_normalize_phone_accepts_mainland_china_formats(raw_phone):
    from app.utils.crypto import normalize_phone

    assert normalize_phone(raw_phone) == "13800138000"


@pytest.mark.parametrize(
    "raw_phone",
    (
        "",
        "12800138000",
        "1380013800",
        "138001380000",
        "+1 202 555 0100",
        "13800138abc",
        "13" + "\uff18" * 9,
        "13" + "\u0660" * 9,
    ),
)
def test_normalize_phone_rejects_invalid_numbers(raw_phone):
    from app.utils.crypto import normalize_phone

    with pytest.raises(ValueError, match="Invalid Chinese mainland phone number"):
        normalize_phone(raw_phone)


def test_phone_encryption_is_randomized_and_round_trips_normalized_number():
    from app.utils.crypto import decrypt_phone, encrypt_phone

    first = encrypt_phone("+86 138-0013-8000")
    second = encrypt_phone("+86 138-0013-8000")

    assert first != second
    assert first.startswith("v1:")
    assert second.startswith("v1:")
    assert decrypt_phone(first) == "13800138000"
    assert decrypt_phone(second) == "13800138000"


def test_phone_decryption_rejects_tampered_ciphertext():
    from app.utils.crypto import decrypt_phone, encrypt_phone

    encrypted = encrypt_phone("13800138000")
    payload = bytearray(base64.b64decode(encrypted.removeprefix("v1:")))
    payload[-1] ^= 1
    tampered = "v1:" + base64.b64encode(payload).decode()

    with pytest.raises(ValueError, match="Invalid encrypted phone"):
        decrypt_phone(tampered)


def test_phone_fingerprint_is_stable_and_keyed(monkeypatch):
    from app.utils import crypto

    monkeypatch.setattr(crypto.settings, "PHONE_HMAC_SECRET", "secret-a")
    first = crypto.fingerprint_phone("+86 138 0013 8000")
    second = crypto.fingerprint_phone("13800138000")

    monkeypatch.setattr(crypto.settings, "PHONE_HMAC_SECRET", "secret-b")
    other_secret = crypto.fingerprint_phone("13800138000")

    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert other_secret != first


def test_mask_phone_hides_middle_digits():
    from app.utils.crypto import mask_phone

    assert mask_phone("+86 138 0013 8000") == "138****8000"


def test_phone_hmac_secret_has_documented_default(monkeypatch):
    from app.config import Settings

    monkeypatch.delenv("PHONE_HMAC_SECRET", raising=False)

    settings = Settings(_env_file=None)

    assert settings.PHONE_HMAC_SECRET == "change-me-phone-hmac-secret"


def test_api_compose_uses_external_phone_security_keys():
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    api = compose.split("  api:", 1)[1].split("  worker:", 1)[0]

    assert "AES_KEY: ${AES_KEY:-change-me-32bytes-secret-key-ok!}" in api
    assert (
        "PHONE_HMAC_SECRET: "
        "${PHONE_HMAC_SECRET:-change-me-phone-hmac-secret}"
    ) in api
