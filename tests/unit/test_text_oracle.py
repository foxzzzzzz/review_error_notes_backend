import json

import httpx
import pytest


def test_preparation_uses_original_answer_but_no_solution(tmp_path):
    from scripts.benchmark_text_oracle import prepare
    source = tmp_path / 'source'
    source.mkdir()
    reference = {'cases': [{'question_id': 'Q1', 'instruction': '看拼音写词语',
        'task_type': 'pinyin_to_characters', 'answer_scope': 'all_slots',
        'reference': {'status': 'assistant_reviewed', 'prompt_text': 'dōng guā',
            'answer_slots': [{'slot_id': 's1', 'state': 'written', 'text': '东'}],
            'correct_answer': 'GOLD_ANSWER', 'explanation_required': 'GOLD_EXPLANATION'}}]}
    (source / 'reference-review.json').write_text(json.dumps(reference), encoding='utf-8')
    (source / 'prepared.json').write_text(json.dumps({'dataset_id': 'fixture'}))
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps({'cases': [{'question_id': 'Q1', 'slot_prompt_texts': {'s1': 'dōng'}}]}))
    prepared = prepare(source, inputs, tmp_path / 'prepared', {'rounds': 3})
    assert prepared['planned_http_attempts'] == 3
    prompt = prepared['requests'][0]['prompt']
    assert '东' in prompt and 'dōng' in prompt
    assert 'GOLD_ANSWER' not in prompt and 'GOLD_EXPLANATION' not in prompt
    reference['cases'][0]['reference']['status'] = 'partial_pending'
    (source / 'reference-review.json').write_text(json.dumps(reference), encoding='utf-8')
    with pytest.raises(ValueError): prepare(source, inputs, tmp_path / 'bad', {'rounds': 3})


@pytest.mark.parametrize('kind', ['valid', 'wrong_id', 'timeout', 'malformed', 'http_error'])
def test_text_call_is_single_attempt_and_preserves_failure(tmp_path, kind):
    from scripts.benchmark_text_oracle import execute_text
    calls = []
    def respond(request):
        calls.append(request)
        if kind == 'timeout': raise httpx.ReadTimeout('secret endpoint token')
        content = json.dumps({'items': [{'question_id': 'Q2' if kind == 'wrong_id' else 'Q1',
            'correct_answer': '冬瓜', 'error_explanation': '冬误写成东', 'uncertain_segments': []}]})
        if kind == 'malformed': content = 'not JSON'
        return httpx.Response(503 if kind == 'http_error' else 200, headers={'x-request-id': 'fixture-id'},
            json={'choices': [{'message': {'content': content}}], 'usage': {'total_tokens': 10}, 'model': 'fixture-model'})
    with httpx.Client(base_url='https://fixture.invalid/v1/', transport=httpx.MockTransport(respond)) as client:
        result, raw = execute_text(client, {'model': 'fixture-model', 'messages': []}, ['Q1'])
    assert len(calls) == result['http_attempts'] == 1
    assert result['status'] == ('parsed' if kind == 'valid' else 'failed')
    assert result['expected_ids'] == ['Q1']
    assert 'secret' not in json.dumps(result) + json.dumps(raw)
    if kind == 'timeout': assert result['transport_error_type'] == 'ReadTimeout'
    if kind == 'valid': assert result['response_ids']['x-request-id'] == 'fixture-id'


def test_text_runner_records_all_rounds_and_actual_model(tmp_path, monkeypatch):
    import hashlib
    from app.config import settings
    from scripts.benchmark_text_oracle import run
    requests = [{'request_id': q, 'expected_ids': [q], 'prompt': q,
        'prompt_sha256': hashlib.sha256(q.encode()).hexdigest(), 'annotation_status': 'assistant-reviewed-text'}
        for q in ['Q1','Q2']]
    prepared = {'scope':'text_solution_oracle','dataset_id':'fixture','requests':requests,
        'expected_questions_per_round':2,'config':{'rounds':3,'timeout_seconds':1,'temperature':0.3,'max_tokens':100}}
    (tmp_path / 'prepared.json').write_text(json.dumps(prepared), encoding='utf-8')
    monkeypatch.setattr(settings, 'LLM_API_KEY', 'fixture-secret')
    monkeypatch.setattr(settings, 'LLM_API_BASE', 'https://fixture.invalid/v1')
    monkeypatch.setattr(settings, 'LLM_MODEL', 'fixture-model')
    original_client = httpx.Client
    calls = []
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        qid = payload['messages'][0]['content']
        if qid == 'Q2': raise httpx.ConnectTimeout('secret exception text')
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'items':[{
            'question_id':qid,'correct_answer':'fixture','error_explanation':'fixture','uncertain_segments':[]}]})}}]})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(respond)))
    output = tmp_path / 'out'
    report = run(tmp_path, output)
    assert report['complete'] and report['http_attempts'] == len(calls) == 6
    assert sum(r['status']=='parsed' for r in report['results']) == 3
    assert len(list(output.glob('*-raw.json'))) == 6
    assert report['runtime']['model'] == 'fixture-model'
    assert 'fixture-secret' not in ''.join(f.read_text(encoding='utf-8') for f in output.glob('*.json'))
