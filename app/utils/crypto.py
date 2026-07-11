from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
from app.config import settings


def encrypt_phone(phone: str) -> str:
    cipher = AES.new(settings.AES_KEY.encode(), AES.MODE_CBC)
    ct = cipher.encrypt(pad(phone.encode(), AES.block_size))
    return base64.b64encode(cipher.iv + ct).decode()


def decrypt_phone(encrypted: str) -> str:
    raw = base64.b64decode(encrypted)
    iv, ct = raw[:16], raw[16:]
    cipher = AES.new(settings.AES_KEY.encode(), AES.MODE_CBC, iv=iv)
    return unpad(cipher.decrypt(ct), AES.block_size).decode()
