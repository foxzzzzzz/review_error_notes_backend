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
    schema = json.loads(prompt.split('\n输出JSON Schema：\n', 1)[1])
    assert schema['properties']['items']['minItems'] == schema['properties']['items']['maxItems'] == 1
    assert schema['$defs']['SolutionItem']['properties']['question_id']['enum'] == ['Q1']
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
        'expected_questions_per_round':2,'config':{'rounds':3,'timeout_seconds':1,'temperature':0.3,'max_tokens':100,
            'expected_endpoint_host':'fixture.invalid','expected_model':'fixture-model','thinking':{'type':'disabled'}}}
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
    assert all(p['thinking'] == {'type': 'disabled'} for p in calls)
    assert report['delivery_counts'] == {'complete': 3, 'failed': 3}
    assert 'fixture-secret' not in ''.join(f.read_text(encoding='utf-8') for f in output.glob('*.json'))


@pytest.mark.parametrize('kind,category', [
    ('length', 'output_truncated'), ('empty', 'empty_final_content'),
    ('duplicate', 'question_id_mismatch'), ('missing_explanation', 'incomplete_solution'),
    ('valid', None),
])
def test_delivery_classifies_observed_server_failures_without_repair(kind, category):
    from scripts.benchmark_text_oracle import execute_text
    item = {'question_id': 'Q1', 'correct_answer': '主席',
            'error_explanation': None if kind == 'missing_explanation' else '行误写，应为席',
            'uncertain_segments': []}
    final = '' if kind in ('length', 'empty') else json.dumps({'items': [item] * (2 if kind == 'duplicate' else 1)})
    data = {'model': 'fixture', 'choices': [{'finish_reason': 'length' if kind == 'length' else 'stop',
            'message': {'content': final, 'reasoning_content': 'not a final answer'}}],
            'usage': {'completion_tokens': 2048, 'completion_tokens_details': {'reasoning_tokens': 2048}}}
    with httpx.Client(base_url='https://fixture.invalid/', transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=data))) as client:
        result, raw = execute_text(client, {'model': 'fixture'}, ['Q1'])
    assert result.get('failure_category') == category
    assert result.get('delivery_status') == ('complete' if kind == 'valid' else 'incomplete' if kind == 'missing_explanation' else 'failed')
    assert result['http_attempts'] == 1 and json.loads(raw['response_body']) == data
    if kind == 'missing_explanation': assert result['status'] == 'parsed'
    if kind in ('length', 'empty', 'duplicate'): assert result['items'] == []


@pytest.mark.parametrize('mismatch', ['host', 'model'])
def test_service_identity_guard_runs_before_any_client_is_created(tmp_path, monkeypatch, mismatch):
    import hashlib
    from app.config import settings
    from scripts.benchmark_text_oracle import run
    prepared = {'scope': 'text_solution_oracle', 'dataset_id': 'fixture', 'expected_questions_per_round': 1,
        'requests': [{'request_id': 'Q1', 'expected_ids': ['Q1'], 'prompt': 'Q1',
                     'prompt_sha256': hashlib.sha256(b'Q1').hexdigest(), 'annotation_status': 'assistant-reviewed-text'}],
        'config': {'rounds': 1, 'timeout_seconds': 1, 'temperature': 0.3, 'max_tokens': 100,
                   'expected_endpoint_host': 'expected.invalid', 'expected_model': 'expected-model'}}
    (tmp_path / 'prepared.json').write_text(json.dumps(prepared), encoding='utf-8')
    monkeypatch.setattr(settings, 'LLM_API_KEY', 'secret')
    monkeypatch.setattr(settings, 'LLM_API_BASE', 'https://other.invalid/v1' if mismatch == 'host' else 'https://expected.invalid/v1')
    monkeypatch.setattr(settings, 'LLM_MODEL', 'other-model' if mismatch == 'model' else 'expected-model')
    clients = []
    def unexpected_client(**kwargs):
        clients.append(kwargs)
        raise RuntimeError('client should never be constructed')
    monkeypatch.setattr(httpx, 'Client', unexpected_client)
    with pytest.raises(ValueError, match='endpoint' if mismatch == 'host' else 'model'):
        run(tmp_path, tmp_path / 'out')
    assert clients == [] and not (tmp_path / 'out').exists()


def test_provider_option_is_explicit_bound_and_never_inferred_for_minimax(monkeypatch):
    from app.config import settings
    from scripts.benchmark_text_oracle import preflight, request_payload
    monkeypatch.setattr(settings, 'LLM_API_BASE', 'https://text.fixture.invalid/v1')
    monkeypatch.setattr(settings, 'LLM_MODEL', 'fixture-text')
    monkeypatch.setattr(settings, 'MINIMAX_API_HOST', 'https://vision.fixture.invalid')
    config = {'temperature': 0.3, 'max_tokens': 2048, 'thinking': {'type': 'disabled'},
              'expected_endpoint_host': 'text.fixture.invalid', 'expected_model': 'fixture-text'}
    identity = preflight(config)
    payload = request_payload('Q', config, 'fixture-text')
    assert payload['thinking'] == {'type': 'disabled'}
    assert identity['text']['request_parameters']['thinking'] == payload['thinking']
    assert identity['vision']['config_source'] == 'MINIMAX_*'
    assert 'thinking' not in identity['vision']
    assert 'thinking' not in request_payload('Q', {'temperature': 0.3, 'max_tokens': 2048}, 'any-model')
    config['expected_endpoint_host'] = None
    with pytest.raises(ValueError, match='bound'):
        preflight(config)


def test_preflight_never_exposes_credentials(monkeypatch):
    from app.config import settings
    from scripts.benchmark_text_oracle import preflight
    monkeypatch.setattr(settings, 'LLM_API_BASE', 'https://user:pass@text.invalid/v1?token=secret')
    monkeypatch.setattr(settings, 'LLM_API_KEY', 'secret-key')
    result = preflight({})
    assert result['text']['endpoint_host'] == 'text.invalid'
    assert all(word not in json.dumps(result) for word in ('secret', 'pass@', 'user:'))
