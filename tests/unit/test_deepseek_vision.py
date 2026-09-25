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
    import re
    from app.config import Settings
    root=Path(__file__).parents[2]
    worker_source = (root/'app/tasks/process_image.py').read_text(encoding='utf-8')
    assert re.search(r'vision_client\s*=\s*create_vision_client\(\)', worker_source)
    assert re.search(r'client\s*=\s*vision_client', worker_source)
    fields=[n for n in Settings.model_fields if n.startswith('DEEPSEEK_VISION_') or n=='VISION_PROVIDER']
    assert fields
    compose=(root/'docker-compose.yml').read_text(encoding='utf-8')
    example=(root/'.env.example').read_text(encoding='utf-8')
    for field in fields:
        assert field+':' in compose and field+'=' in example


def test_default_configuration_creates_deepseek_with_text_key(monkeypatch):
    from app.config import Settings
    from app.services import vision_provider as provider
    from app.services import deepseek_vision
    config = Settings(_env_file=None, LLM_API_KEY='test-only', LLM_API_BASE='https://api.deepseek.com/v1')
    monkeypatch.setattr(provider, 'settings', config)
    monkeypatch.setattr(deepseek_vision, 'settings', config)
    client = provider.create_vision_client()
    assert isinstance(client, deepseek_vision.DeepSeekVisionClient)
    assert client.api_key == 'test-only'
    config.LLM_API_KEY = ''
    with pytest.raises(ValueError, match='credential'):
        provider.create_vision_client()


def test_deepseek_localized_content_uses_source_aware_prompt_and_parses_observations(tmp_path):
    image = tmp_path / 'crop.png'
    Image.new('RGB', (160, 240), 'white').save(image)
    captured = {}
    content = {
        'items': [{
            'mark_id': 4,
            'raw_text': 'qing ting',
            'instruction': '看词语写拼音',
            'prompt_text': '蜻蜓',
            'normalized_text': None,
            'answer': None,
            'subject': 'chinese',
            'question_type': 'write_pinyin',
            'tags': [],
            'difficulty': 2,
            'confidence': 0.9,
            'uncertain_segments': [],
            'student_handwriting': {
                'source': 'deepseek',
                'source_class': 'student_handwriting',
                'text': 'qing ting',
                'bbox': [0.2, 0.3, 0.5, 0.4],
                'confidence': 0.88,
            },
        }]
    }

    def respond(request):
        captured['body'] = json.loads(request.content)
        return httpx.Response(200, json={
            'model': 'deepseek-flash',
            'choices': [
                {'message': {'content': json.dumps(content)}, 'finish_reason': 'stop'}
            ],
        })

    result = deepseek_client(respond).recognize_localized_content(str(image), [4])

    prompt = captured['body']['messages'][0]['content'][0]['text']
    assert '学生作答格中的文字按 student_handwriting 输出' in prompt
    assert 'teacher_correction' in prompt
    assert result.items[0].mark_id == 4
    assert result.items[0].student_handwriting.text == 'qing ting'


def test_deepseek_marked_page_uses_one_standardized_page_and_external_prompt(tmp_path):
    image = tmp_path / 'page.png'
    Image.new('RGB', (3200, 1600), 'white').save(image)
    requests = []
    response_content = {
        'wrong_questions': [{
            'printed_question': '看图填空',
            'student_answer': None,
            'model_bbox': [0.1, 0.2, 0.4, 0.5],
            'confidence': 0.91,
            'uncertain_fields': ['student_answer'],
            'correct_answer_suggestion': '正确答案',
        }],
    }

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            'choices': [{
                'message': {'content': json.dumps(response_content)},
                'finish_reason': 'stop',
            }],
        })

    result = deepseek_client(respond).recognize_marked_page(str(image))

    assert len(requests) == 1
    body = requests[0]
    content = body['messages'][0]['content']
    assert len(content) == 2
    assert content[1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    prompt = content[0]['text']
    assert "纠偏要求：" not in prompt
    assert 'model_bbox' in prompt and 'uncertain_fields' in prompt
    assert '每个被老师批改的答题格、填空位或选择项作为一个最小独立候选' in prompt
    assert '同一行多个批改项必须分别返回，禁止整行合并' in prompt
    assert '教师批改痕迹使用红色笔迹' in prompt
    assert '红圈和红叉是错题定位依据' in prompt
    assert '红勾表示答案正确，不能据此返回错题候选' in prompt
    assert '其他红色文字、订正内容、划线或零散笔迹不能单独作为错题依据' in prompt
    assert '黑色、灰色或其他非红色笔迹' in prompt
    assert 'teacher_mark_type' not in prompt and 'teacher_mark_bbox' not in prompt
    assert '返回候选数量应与' not in prompt
    assert '其他明确批改标记' not in prompt
    for forbidden in ('OCR', 'CV', 'region_id', 'sample_id', '人工答案'):
        assert forbidden not in prompt
    assert 'correct_answer_suggestion' in prompt
    assert result.wrong_questions[0].student_answer is None
    assert result.wrong_questions[0].correct_answer_suggestion == '正确答案'


@pytest.mark.parametrize(
    ("correction", "expected_instruction"),
    [
        ("missed_errors", "本图上次可能漏识别错题"),
        ("false_positives", "本图上次可能误识别正确题"),
        ("both", "本图上次可能同时漏识别错题并误识别正确题"),
    ],
)
def test_deepseek_marked_page_appends_selected_correction_instruction(
    tmp_path, correction, expected_instruction
):
    image = tmp_path / "page.png"
    Image.new("RGB", (320, 240), "white").save(image)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": json.dumps({"wrong_questions": []})},
                "finish_reason": "stop",
            }],
        })

    deepseek_client(respond).recognize_marked_page(str(image), correction=correction)

    prompt = requests[0]["messages"][0]["content"][0]["text"]
    assert "\n\n纠偏要求：" + expected_instruction in prompt


def test_deepseek_marked_page_preserves_exact_production_response_content_for_audit(tmp_path):
    image = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(image)
    raw_content = "```json\n" + json.dumps({"wrong_questions": []}) + "\n```"

    result = deepseek_client(
        lambda _request: httpx.Response(200, json={
            "choices": [{"message": {"content": raw_content}, "finish_reason": "stop"}],
        })
    ).recognize_marked_page(str(image))

    assert result.raw_response_content == raw_content


def test_deepseek_marked_page_format_error_makes_exactly_one_call(tmp_path):
    image = tmp_path / 'page.png'
    Image.new('RGB', (160, 240), 'white').save(image)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        content = (
            '{"wrong_questions":'
            if len(requests) == 1
            else json.dumps({'wrong_questions': []})
        )
        return httpx.Response(200, json={
            'choices': [{
                'message': {'content': content},
                'finish_reason': 'stop',
            }],
        })

    client = deepseek_client(respond)
    client.max_retries = 1
    with pytest.raises(VisionRecognitionError, match="格式"):
        client.recognize_marked_page(str(image))

    assert len(requests) == 1
