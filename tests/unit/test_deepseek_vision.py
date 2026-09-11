import json

import httpx
import pytest
from PIL import Image

from app.services.vision_recognition import MiniMaxVisionClient, VisionRecognitionError


def deepseek_client(handler):
    from app.services.deepseek_vision import DeepSeekVisionClient
    return DeepSeekVisionClient(api_key='deepseek-test-key', api_host='https://api.deepseek.com/v1',
        model='deepseek-flash', thinking='disabled', max_tokens=4096, image_detail='original',
        timeout_seconds=20, max_retries=0, max_edge=2048, jpeg_quality=90,
        transport=httpx.MockTransport(handler))


def test_detect_marks_preserves_same_prompt_and_image_as_minimax(tmp_path):
    image = tmp_path/'page.png'
    Image.new('RGB',(160,240),'white').save(image)
    requests = []
    def respond(request):
        body=json.loads(request.content)
        requests.append((request,body))
        content=json.dumps({'error_marks':[]})
        if 'messages' in body:
            return httpx.Response(200,json=dict(model='deepseek-flash',choices=[dict(message=dict(content=content),finish_reason='stop')],usage=dict(total_tokens=7)))
        return httpx.Response(200,json=dict(base_resp=dict(status_code=0),content=content))
    mini=MiniMaxVisionClient(api_key='mini-test-key',api_host='https://api.minimaxi.com',timeout_seconds=20,
        max_retries=0,max_edge=2048,jpeg_quality=90,transport=httpx.MockTransport(respond))
    ds=deepseek_client(respond)
    assert mini.detect_marks(str(image),[[.1,.2,.3,.4]]) == ds.detect_marks(str(image),[[.1,.2,.3,.4]])
    m,d=requests[0][1],requests[1][1]
    assert d['messages'][0]['content'][0]==dict(type='text',text=m['prompt'])
    assert d['messages'][0]['content'][1]['image_url']['url']==m['image_url']
    assert d['model']=='deepseek-flash' and d['thinking']=={'type':'disabled'}
    assert str(requests[1][0].url)=='https://api.deepseek.com/v1/chat/completions'
    assert requests[1][0].headers['authorization']=='Bearer deepseek-test-key'
    assert 'mm-api-source' not in requests[1][0].headers and 'response_format' not in d


@pytest.mark.parametrize('body,code',[
    ({'choices':[]},'vision_response_envelope_invalid'),
    ({'choices':[{'message':{'content':'{}'},'finish_reason':'length'}]},'vision_response_truncated'),
    ({'choices':[{'message':{'reasoning_content':'{}','content':None},'finish_reason':'stop'}]},'vision_response_empty'),
])
def test_deepseek_invalid_results_are_not_empty_success(body,code):
    from scripts.local_vision_capability import Decision
    client=deepseek_client(lambda request:httpx.Response(200,json=body))
    with pytest.raises(VisionRecognitionError) as error:
        client._request(dict(prompt='same',image_url='data:image/png;base64,eA=='),Decision,{'operation':'test'})
    assert error.value.code==code


def test_factory_defaults_to_minimax_and_never_falls_back(monkeypatch):
    from app.services import vision_provider as provider
    monkeypatch.setattr(provider.settings,'VISION_PROVIDER','minimax')
    assert type(provider.create_vision_client()) is MiniMaxVisionClient
    monkeypatch.setattr(provider.settings,'DEEPSEEK_VISION_API_KEY','')
    monkeypatch.setattr(provider.settings,'DEEPSEEK_VISION_KEY_SOURCE','dedicated')
    monkeypatch.setattr(provider.settings,'LLM_API_KEY','text-secret')
    with pytest.raises(ValueError,match='credential'):
        provider.create_vision_client('deepseek')
    monkeypatch.setattr(provider.settings,'DEEPSEEK_VISION_KEY_SOURCE','text_llm')
    monkeypatch.setattr(provider.settings,'LLM_API_BASE','https://api.openai.com/v1')
    with pytest.raises(ValueError,match='host'):
        provider.create_vision_client('deepseek')
    monkeypatch.setattr(provider.settings,'LLM_API_BASE','https://api.deepseek.com/v1')
    client=provider.create_vision_client('deepseek')
    assert client.api_key=='text-secret' and client.model=='deepseek-flash'


def test_worker_uses_provider_factory_and_compose_passes_settings():
    from pathlib import Path
    from app.config import Settings
    root=Path(__file__).parents[2]
    assert 'client=create_vision_client()' in (root/'app/tasks/process_image.py').read_text(encoding='utf-8')
    fields=[n for n in Settings.model_fields if n.startswith('DEEPSEEK_VISION_') or n=='VISION_PROVIDER']
    assert fields
    compose=(root/'docker-compose.yml').read_text(encoding='utf-8')
    example=(root/'.env.example').read_text(encoding='utf-8')
    for field in fields:
        assert field+':' in compose and field+'=' in example
