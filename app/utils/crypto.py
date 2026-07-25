import base64
import binascii
import hashlib
import hmac
import re

from Crypto.Cipher import AES

from app.config import settings


PHONE_CIPHERTEXT_VERSION = "v1"
PHONE_GCM_NONCE_BYTES = 16
PHONE_GCM_TAG_BYTES = 16


def normalize_phone(phone: str) -> str:
    compact = re.sub(r"[\s-]+", "", phone or "")
    if compact.startswith("+86"):
        compact = compact[3:]
    elif compact.startswith("86") and len(compact) == 13:
        compact = compact[2:]
    if re.fullmatch(r"1[3-9][0-9]{9}", compact) is None:
        raise ValueError("Invalid Chinese mainland phone number")
    return compact


def encrypt_phone(phone: str) -> str:
    normalized = normalize_phone(phone)
    cipher = AES.new(settings.AES_KEY.encode(), AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(normalized.encode("ascii"))
    payload = cipher.nonce + tag + ciphertext
    return f"{PHONE_CIPHERTEXT_VERSION}:{base64.b64encode(payload).decode()}"


def decrypt_phone(encrypted: str) -> str:
    prefix = f"{PHONE_CIPHERTEXT_VERSION}:"
    if not encrypted.startswith(prefix):
        raise ValueError("Invalid encrypted phone")
    try:
        raw = base64.b64decode(encrypted[len(prefix):], validate=True)
        minimum_size = PHONE_GCM_NONCE_BYTES + PHONE_GCM_TAG_BYTES + 1
        if len(raw) < minimum_size:
            raise ValueError("Invalid encrypted phone")
        nonce = raw[:PHONE_GCM_NONCE_BYTES]
        tag = raw[
            PHONE_GCM_NONCE_BYTES:
            PHONE_GCM_NONCE_BYTES + PHONE_GCM_TAG_BYTES
        ]
        ciphertext = raw[PHONE_GCM_NONCE_BYTES + PHONE_GCM_TAG_BYTES:]
        cipher = AES.new(settings.AES_KEY.encode(), AES.MODE_GCM, nonce=nonce)
        phone = cipher.decrypt_and_verify(ciphertext, tag).decode("ascii")
        return normalize_phone(phone)
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Invalid encrypted phone") from exc


def fingerprint_phone(phone: str) -> str:
    normalized = normalize_phone(phone)
    return hmac.new(
        settings.PHONE_HMAC_SECRET.encode("utf-8"),
        normalized.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def mask_phone(phone: str) -> str:
    normalized = normalize_phone(phone)
    return f"{normalized[:3]}****{normalized[-4:]}"
