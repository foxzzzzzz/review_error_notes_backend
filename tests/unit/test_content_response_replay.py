import json
import pytest

from scripts.replay_content_responses import audit_response


def payload():
    return {'items': [{'question_id': 'Q1', 'prompt_text': 'dōng guā', 'student_answer': '东瓜',
        'correct_answer': '冬瓜', 'error_explanation': '将"东"改为"冬"。', 'uncertain_segments': []}]}


def test_valid_response_is_never_rewritten():
    raw = json.dumps(payload(), ensure_ascii=False)
    result = audit_response(raw, ['Q1'])
    assert result['status'] == 'strict_valid'
    assert result['candidate_json'] == raw
    assert result['inserted_backslash_positions'] == []


def test_quote_repair_only_inserts_backslashes_preserving_all_content():
    raw = json.dumps(payload(), ensure_ascii=False).replace('\\"', '"')
    result = audit_response('```json\n' + raw + '\n```', ['Q1'])
    assert result['status'] == 'quote_candidate_pending_review'
    assert json.loads(result['candidate_json']) == payload()
    rebuilt = raw
    for pos in reversed(result['inserted_backslash_positions']):
        rebuilt = rebuilt[:pos] + '\\' + rebuilt[pos:]
    assert rebuilt == result['candidate_json']
    assert result['production_accepted'] is False


@pytest.mark.parametrize('kind', ['missing_field', 'bracket', 'wrong_id', 'embedded_field', 'duplicate_key'])
def test_damaged_or_ambiguous_response_is_not_repaired(kind):
    data = payload()
    if kind == 'missing_field':
        del data['items'][0]['uncertain_segments']
    elif kind == 'wrong_id':
        data['items'][0]['question_id'] = 'Q2'
    elif kind == 'embedded_field':
        data['items'][0]['error_explanation'] = '文字含 "correct_answer": "不可修改"'
    raw = json.dumps(data, ensure_ascii=False).replace('\\"', '"')
    if kind == 'bracket':
        raw = raw[:-2] + ']'
    if kind == 'duplicate_key':
        raw = raw.replace('"correct_answer":', '"correct_answer": "伪造", "correct_answer":')
    assert audit_response(raw, ['Q1'])['status'] == 'unrecoverable'


def test_existing_backslashes_and_quotes_are_preserved():
    data = payload()
    data['items'][0]['student_answer'] = '原文\\路径'
    raw = json.dumps(data, ensure_ascii=False).replace('将\\"东\\"', '将"东"')
    result = audit_response(raw, ['Q1'])
    assert json.loads(result['candidate_json']) == data
