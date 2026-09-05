import pytest
import json
import httpx

from scripts.benchmark_content_oracle import ContentResult, validate_ids, execute_request, run
from scripts.prepare_convergence_dataset import sha256


def item(qid):
    return {'question_id': qid, 'prompt_text': '题干', 'student_answer': '原错答',
            'correct_answer': '正答', 'error_explanation': '可观察错误', 'uncertain_segments': []}


def test_valid_content_preserves_original_answer():
    result = ContentResult.model_validate({'items': [item('Q1')]})
    validate_ids(result, ['Q1'])
    assert result.items[0].student_answer == '原错答'


@pytest.mark.parametrize('ids', [['Q1', 'Q1'], ['unknown'], []])
def test_invalid_ids_rejected(ids):
    result = ContentResult.model_validate({'items': [item(qid) for qid in ids]})
    with pytest.raises(ValueError):
        validate_ids(result, ['Q1'])


@pytest.mark.parametrize('failure', [TimeoutError(), ValueError('malformed')])
def test_failure_retains_all_expected_questions(failure):
    class Client:
        def _request(self, *args):
            raise failure
    result = execute_request(Client(), {'prompt': 'test'}, ['Q1', 'Q2'], ContentResult)
    assert result['expected_ids'] == ['Q1', 'Q2']
    assert result['status'] == 'failed'
    assert result['items'] == []
    assert result['elapsed_ms'] >= 0


def test_explanation_is_required():
    row = item('Q1')
    del row['error_explanation']
    with pytest.raises(ValueError):
        ContentResult.model_validate({'items': [row]})


@pytest.mark.parametrize('response_kind', ['valid', 'malformed', 'timeout'])
def test_server_runner_counts_actual_attempts_without_retry(tmp_path, monkeypatch, response_kind):
    from app.services.vision_recognition import MiniMaxVisionClient
    calls = []
    def respond(request):
        calls.append(request)
        if response_kind == 'timeout':
            raise httpx.ReadTimeout('fixture')
        content = json.dumps({'items': [item('Q1')]}) if response_kind == 'valid' else 'not JSON'
        return httpx.Response(200, json={'content': content})
    client = MiniMaxVisionClient('fixture-secret', 'https://fixture.invalid', 1, 2, 2048, 90,
                                transport=httpx.MockTransport(respond))
    monkeypatch.setattr(MiniMaxVisionClient, 'from_settings', classmethod(lambda cls: client))
    (tmp_path / 'image.jpg').write_bytes(b'fixture-image')
    prepared = {'scope': 'content_oracle', 'dataset_id': 'fixture', 'expected_questions_per_round': 1,
        'config': {'content_timeout_seconds': 1, 'content_rounds': 2},
        'requests': [{'annotation_status': 'region-verified', 'image': 'image.jpg',
            'image_sha256': sha256(tmp_path / 'image.jpg'), 'expected_ids': ['Q1'],
            'prompt': 'fixture', 'request_id': 'one', 'label': 'page'}]}
    (tmp_path / 'prepared.json').write_text(json.dumps(prepared), encoding='utf-8')
    output = tmp_path / 'out'
    report = run(tmp_path, output)
    assert len(calls) == report['http_attempts'] == 2
    assert all(r['expected_ids'] == ['Q1'] for r in report['results'])
    assert all(r['status'] == ('parsed' if response_kind == 'valid' else 'failed') for r in report['results'])
    assert report['review_status'] == 'pending'
    assert 'fixture-secret' not in ''.join(p.read_text(encoding='utf-8') for p in output.glob('*.json'))
