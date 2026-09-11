import json
from pathlib import Path

import httpx
import pytest
from PIL import Image


def test_paired_execution_preserves_inputs_and_failures(tmp_path):
    from scripts.local_vision_capability import prepare,sha,read,check_results
    from scripts.vision_provider_comparison import run_pair,compare
    from app.services.deepseek_vision import DeepSeekVisionClient
    from app.services.vision_recognition import MiniMaxVisionClient
    image=tmp_path/'input.png'
    Image.new('RGB',(180,240),'white').save(image)
    folder=prepare([dict(source_id='private',source_image=str(image),source_sha256=sha(image),bbox=[.1,.2,.8,.7],
                        group='old_positive',expected='wrong',expected_word='字',expected_marked_text='字',protected=True)],
                   Path(__file__).parents[2]/'scripts/local_vision_capability_config.json',tmp_path/'input')
    sent=[]
    def respond(request):
        body=json.loads(request.content);provider='deepseek' if 'messages' in body else 'minimax'
        sent.append((provider,body))
        if len(sent)==3:
            return httpx.Response(503,json={})
        content=json.dumps(dict(case_id='V001',verdict='wrong',targets=[dict(word='字',marked_text='字')],evidence='red cross'))
        result=dict(choices=[dict(message=dict(content=content),finish_reason='stop')],model='deepseek-flash') if provider=='deepseek' else dict(content=content,base_resp=dict(status_code=0))
        return httpx.Response(200,json=result)
    kwargs=dict(api_key='test',timeout_seconds=20,max_retries=3,max_edge=2048,jpeg_quality=90,transport=httpx.MockTransport(respond))
    clients={'minimax':MiniMaxVisionClient(api_host='https://api.minimaxi.com',**kwargs),
             'deepseek':DeepSeekVisionClient(api_host='https://api.deepseek.com/v1',model='deepseek-flash',thinking='disabled',max_tokens=4096,image_detail='original',**kwargs)}
    config=dict(providers=['minimax','deepseek'],max_total_attempts=4)
    run_pair(clients,folder,tmp_path/'result',config,{p:{'provider':p} for p in clients})
    assert [p for p,b in sent]==['minimax','deepseek','deepseek','minimax']
    for _,body in sent:
        if 'messages' in body:
            assert body['messages'][0]['content'][0]['text']==sent[0][1]['prompt']
            assert body['messages'][0]['content'][1]['image_url']['url']==sent[0][1]['image_url']
    for provider in clients:
        assert check_results(folder,tmp_path/'result'/provider)['integrity_complete']
    scores=compare(folder,tmp_path/'result',tmp_path/'input-labels.json')
    assert scores['complete'] and len(scores['paired_cases'])==2
    assert scores['paired_cases'][1]['minimax']['classification_correct']
    assert not scores['paired_cases'][1]['deepseek']['classification_correct']
    assert scores['provider_scores']['deepseek']['groups']['2:old_positive']['total']==1


def test_pair_budget_checked_before_creating_output(tmp_path):
    from scripts.vision_provider_comparison import validate_budget
    with pytest.raises(ValueError,match='budget'):
        validate_budget(dict(schedule=[{},{}]),dict(providers=['minimax','deepseek'],max_total_attempts=3))
