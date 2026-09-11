"""DeepSeek transport for the existing visual prompts, stages and validation."""
from urllib.parse import urlsplit

import httpx

from app.config import settings
from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError, _extract_json


class DeepSeekVisionClient(MiniMaxVisionClient):
    def __init__(self, *, model, thinking, max_tokens, image_detail, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.image_detail = image_detail

    @classmethod
    def from_settings(cls):
        key = settings.DEEPSEEK_VISION_API_KEY
        if settings.DEEPSEEK_VISION_KEY_SOURCE == 'text_llm':
            if urlsplit(settings.LLM_API_BASE).hostname != urlsplit(settings.DEEPSEEK_VISION_API_BASE).hostname:
                raise ValueError('text LLM host differs from DeepSeek vision host')
            key = settings.LLM_API_KEY
        if not key:
            raise ValueError('DeepSeek vision credential is not configured')
        return cls(api_key=key, api_host=settings.DEEPSEEK_VISION_API_BASE,
                   model=settings.DEEPSEEK_VISION_MODEL, thinking=settings.DEEPSEEK_VISION_THINKING,
                   max_tokens=settings.DEEPSEEK_VISION_MAX_TOKENS, image_detail=settings.DEEPSEEK_VISION_IMAGE_DETAIL,
                   timeout_seconds=settings.DEEPSEEK_VISION_TIMEOUT_SECONDS,
                   max_retries=settings.DEEPSEEK_VISION_MAX_RETRIES,
                   max_edge=settings.MINIMAX_IMAGE_MAX_EDGE, jpeg_quality=settings.MINIMAX_IMAGE_JPEG_QUALITY,
                   retry_delay_seconds=settings.MINIMAX_VISION_RETRY_DELAY_SECONDS)

    def _post(self, payload):
        request = dict(model=self.model, thinking={'type':self.thinking}, max_tokens=self.max_tokens,
                       messages=[dict(role='user', content=[
                           dict(type='text', text=payload['prompt']),
                           dict(type='image_url', image_url=dict(url=payload['image_url'], detail=self.image_detail)),
                       ])])
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            return client.post(self.api_host+'/chat/completions',
                               headers={'Authorization':'Bearer '+self.api_key,'Content-Type':'application/json'},
                               json=request)

    @staticmethod
    def _parse_response_content(response, diagnostic):
        try:
            data = response.json()
            choices = data['choices']
            if not isinstance(choices,list) or len(choices) != 1:
                raise ValueError('expected one choice')
            choice = choices[0]
            content = choice['message'].get('content')
            finish = choice.get('finish_reason')
            if content is not None and not isinstance(content,str):
                raise ValueError('expected text content')
        except (ValueError,KeyError,TypeError,AttributeError) as exc:
            raise VisionRecognitionError('vision_response_envelope_invalid','视觉服务返回格式异常',diagnostic) from exc
        if finish == 'length':
            raise VisionRecognitionError('vision_response_truncated','视觉服务输出被截断',diagnostic)
        if finish not in (None,'stop'):
            raise VisionRecognitionError('vision_upstream_rejected','视觉服务未正常完成识别',diagnostic)
        # Never substitute reasoning_content for the final structured answer.
        return _extract_json(content or '',diagnostic)
