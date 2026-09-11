"""Explicit vision provider selection; no automatic provider fallback."""
from app.config import settings
from app.services.deepseek_vision import DeepSeekVisionClient
from app.services.vision_recognition import MiniMaxVisionClient


def create_vision_client(provider=None):
    provider = provider or settings.VISION_PROVIDER
    if provider == 'minimax':
        return MiniMaxVisionClient.from_settings()
    if provider == 'deepseek':
        return DeepSeekVisionClient.from_settings()
    raise ValueError('unsupported vision provider')
