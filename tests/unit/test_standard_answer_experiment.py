import copy
import json
from pathlib import Path

import httpx
import pytest


def config():
    return json.loads((Path(__file__).parents[2] / 'scripts/standard_answer_config.json').read_text(encoding='utf-8'))


def case(original=None, task='pinyin_to_characters'):
    return {'question_id': 'Q1', 'task_type': task,
            'answer_slots': original or [{'slot_id': 's1', 'state': 'written', 'text': '秋'},
                                        {'slot_id': 's2', 'state': 'blank', 'text': ''}]}


def answer(first='蚯', second='蚓'):
    return {'question_id': 'Q1', 'standard_slots': [{'slot_id': 's2', 'text': second},
            {'slot_id': 's1', 'text': first}], 'uncertain_segments': []}


def test_comparison_covers_every_wrong_slot_without_changing_original():
    from scripts.standard_answer_experiment import compare_slots
    original = case()
    untouched = copy.deepcopy(original)
    result = compare_slots(original, answer(), config())
    assert original == untouched
    assert result['question_status'] == 'incorrect'
    assert result['correct_answer'] == '蚯蚓'
    assert [r['status'] for r in result['slot_comparisons']] == ['wrong', 'missing']
    assert all(word in result['error_explanation'] for word in ['秋', '蚯', '蚓', '未作答'])


def test_comparison_preserves_tones_and_accepts_unicode_equivalent_tone():
    from scripts.standard_answer_experiment import compare_slots
    original = case([{'slot_id': 's1', 'state': 'written', 'text': 'tān'},
                     {'slot_id': 's2', 'state': 'written', 'text': 'tu\u030c'}], task='characters_to_pinyin')
    result = compare_slots(original, answer('tán', 'tǔ'), config())
    assert [s['status'] for s in result['slot_comparisons']] == ['wrong', 'match']
    assert result['correct_answer'] == 'tán tǔ'


@pytest.mark.parametrize('uncertainty', ['student', 'standard', 'model'])
def test_uncertain_evidence_never_becomes_complete_or_blank(uncertainty):
    from scripts.standard_answer_experiment import compare_slots
    original, predicted = case(), answer()
    if uncertainty == 'student': original['answer_slots'][0].update(state='uncertain', text=None)
    if uncertainty == 'standard': predicted['standard_slots'][1]['text'] = None
    if uncertainty == 'model': predicted['uncertain_segments'] = ['题面不清楚']
    result = compare_slots(original, predicted, config())
    assert result['question_status'] == 'needs_review'
    assert result['comparison_complete'] is False


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'unknown', 'question'])
def test_slot_ids_are_not_repaired_or_inferred(mutation):
    from scripts.standard_answer_experiment import compare_slots
    predicted = answer()
    if mutation == 'duplicate': predicted['standard_slots'][1]['slot_id'] = 's2'
    if mutation == 'missing': predicted['standard_slots'].pop()
    if mutation == 'unknown': predicted['standard_slots'][1]['slot_id'] = 's3'
    if mutation == 'question': predicted['question_id'] = 'Q2'
    with pytest.raises(ValueError): compare_slots(case(), predicted, config())


def test_correct_student_and_target_only_are_not_false_positives():
    from scripts.standard_answer_experiment import compare_slots
    original = case([{'slot_id': 's1', 'state': 'written', 'text': '运'}], task='pinyin_sentence')
    predicted = {'question_id': 'Q1', 'standard_slots': [{'slot_id': 's1', 'text': '运'}], 'uncertain_segments': []}
    result = compare_slots(original, predicted, config())
    assert result['question_status'] == 'correct' and result['error_explanation'] is None


def test_prepare_separates_student_and_gold_from_solver_input(tmp_path):
    from scripts.standard_answer_experiment import prepare
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'prepared.json').write_text(json.dumps({'dataset_id': 'fixture'}))
    original = case([{'slot_id': 's1', 'state': 'written', 'text': 'STUDENT_SENTINEL'},
                     {'slot_id': 's2', 'state': 'blank', 'text': ''}])
    ref = {'question_id': 'Q1', 'task_type': 'pinyin_to_characters', 'answer_scope': 'all_slots',
           'reference': {'status': 'assistant_reviewed', 'prompt_text': 'qiū yǐn',
             'answer_slots': original['answer_slots'], 'correct_answer': 'GOLD_SENTINEL', 'explanation_required': 'EXPLANATION_SENTINEL'}}
    (source / 'reference-review.json').write_text(json.dumps({'cases': [ref]}), encoding='utf-8')
    cues = tmp_path / 'cues.json'
    cues.write_text(json.dumps({'cases': [{'question_id': 'Q1', 'slot_prompt_texts': {'s1': 'qiū', 's2': 'yǐn'}}]}), encoding='utf-8')
    out = tmp_path / 'prepared'
    prepared = prepare(source, cues, out, config())
    request = prepared['requests'][0]
    assert all(s not in request['prompt'] for s in ['STUDENT_SENTINEL', 'GOLD_SENTINEL', 'EXPLANATION_SENTINEL', 'blank'])
    assert 'qiū' in request['prompt'] and 'yǐn' in request['prompt']
    assert 'STUDENT_SENTINEL' in (out / 'student-inputs.json').read_text(encoding='utf-8')
    assert 'GOLD_SENTINEL' not in (out / 'student-inputs.json').read_text(encoding='utf-8')


@pytest.mark.parametrize('kind', ['plain', 'fence', 'prefix', 'length', 'duplicate', 'timeout'])
def test_single_attempt_with_bounded_format_handling(kind):
    from scripts.standard_answer_experiment import execute_standard
    value = answer()
    if kind == 'duplicate': value['standard_slots'][1]['slot_id'] = 's2'
    content = json.dumps(value)
    if kind == 'fence': content = '```json\n' + content + '\n```'
    if kind == 'prefix': content = 'Answer: ' + content
    def respond(request):
        if kind == 'timeout': raise httpx.ReadTimeout('secret endpoint')
        return httpx.Response(200, json={'model': 'fixture-model', 'choices': [{
            'finish_reason': 'length' if kind == 'length' else 'stop',
            'message': {'content': content, 'reasoning_content': 'NEVER_USE_THIS'}}]})
    with httpx.Client(base_url='https://fixture.invalid/', transport=httpx.MockTransport(respond)) as client:
        result, raw = execute_standard(client, {'model': 'fixture-model'}, case(), config())
    assert result['http_attempts'] == 1
    assert result['status'] == ('parsed' if kind in ('plain', 'fence') else 'failed')
    if kind == 'fence':
        assert result['format_normalization'] == 'outer_json_fence'
        assert '```json' in raw['response_body']
    assert 'NEVER_USE_THIS' not in json.dumps(result)
    if kind == 'timeout': assert 'secret' not in json.dumps(result) + json.dumps(raw)


def test_full_run_binds_input_and_keeps_failures_without_student_leak(tmp_path, monkeypatch):
    import hashlib
    from app.config import settings
    from scripts.standard_answer_experiment import run
    students = {'cases': [case()]}
    path = tmp_path / 'student-inputs.json'
    path.write_text(json.dumps(students), encoding='utf-8')
    cfg = config()
    cfg.update(expected_endpoint_host='fixture.invalid', expected_model='fixture-model')
    prompt = 'SOLVE_PROBLEM_ONLY'
    prepared = {'scope': 'standard_answer_slots', 'config': cfg,
        'student_inputs_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'requests': [{'request_id': 'Q1', 'expected_slot_ids': ['s1', 's2'], 'prompt': prompt,
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}]}
    (tmp_path / 'prepared.json').write_text(json.dumps(prepared), encoding='utf-8')
    monkeypatch.setattr(settings, 'LLM_API_BASE', 'https://fixture.invalid/v1')
    monkeypatch.setattr(settings, 'LLM_MODEL', 'fixture-model')
    monkeypatch.setattr(settings, 'LLM_API_KEY', 'SECRET')
    calls = []
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload['messages'] == [{'role': 'user', 'content': prompt}]
        assert payload['thinking'] == {'type': 'disabled'}
        if len(calls) == 2: raise httpx.ConnectTimeout('SECRET')
        return httpx.Response(200, json={'model': 'fixture-model', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': json.dumps(answer())}}]})
    original_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(respond)))
    result = run(tmp_path, tmp_path / 'out')
    assert len(calls) == result['http_attempts'] == 3
    assert result['complete'] and result['delivery_counts'] == {'complete': 2, 'failed': 1}
    assert len(list((tmp_path / 'out').glob('*-raw.json'))) == 3
    assert 'SECRET' not in ''.join(f.read_text(encoding='utf-8') for f in (tmp_path / 'out').glob('*.json'))
    students['cases'][0]['answer_slots'][0]['text'] = 'mutated'
    path.write_text(json.dumps(students), encoding='utf-8')
    with pytest.raises(ValueError, match='hash'): run(tmp_path, tmp_path / 'bad')
    assert len(calls) == 3 and not (tmp_path / 'bad').exists()


def test_model_identity_change_cannot_be_counted_as_solver_success():
    from scripts.standard_answer_experiment import execute_standard
    def respond(request):
        return httpx.Response(200, json={'model': 'different-model', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': json.dumps(answer())}}]})
    with httpx.Client(base_url='https://fixture.invalid/', transport=httpx.MockTransport(respond)) as client:
        result, _ = execute_standard(client, {'model': 'fixture-model'}, case(), config())
    assert result['failure_category'] == 'model_mismatch' and result['status'] == 'failed'
