import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.content_context_experiment import FullContentResult, TranscriptionResult, prepare, validate_slots
from scripts.prepare_convergence_dataset import sha256


def transcription():
    return {'question_id': 'Q1', 'prompt_text': 'dōng guā',
            'answer_slots': [{'slot_id': 's1', 'state': 'written', 'text': '东'},
                             {'slot_id': 's2', 'state': 'blank', 'text': ''}], 'uncertain_segments': []}


def test_transcription_cannot_sneak_in_correct_answers():
    row = dict(transcription(), correct_answer='冬瓜')
    with pytest.raises(ValueError):
        TranscriptionResult.model_validate({'items': [row]})
    parsed = FullContentResult.model_validate({'items': [dict(row, error_explanation='错字及漏答')]})
    validate_slots(parsed, {'Q1': ['s1', 's2']})


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'unknown', 'fake_blank'])
def test_slot_identity_and_blank_semantics_are_enforced(kind):
    row = transcription()
    if kind == 'duplicate': row['answer_slots'][1]['slot_id'] = 's1'
    if kind == 'missing': row['answer_slots'].pop()
    if kind == 'unknown': row['answer_slots'][1]['slot_id'] = 's3'
    if kind == 'fake_blank': row['answer_slots'][1]['text'] = '猜测的字'
    with pytest.raises(ValueError):
        parsed = TranscriptionResult.model_validate({'items': [row]})
        validate_slots(parsed, {'Q1': ['s1', 's2']})


def test_both_arms_share_images_context_and_never_send_gold(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    Image.new('RGB', (100, 100), 'white').save(dataset / 'page.png')
    manifest = {'dataset_id': 'fixture', 'pages': [{'label': 'page', 'image': 'page.png',
        'image_sha256': sha256(dataset / 'page.png'),
        'questions': [{'question_id': 'Q1', 'bbox': [0, 0, 1, 1]}]}]}
    (dataset / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    cases = {'version': 'fixture', 'cases': [{'question_id': 'Q1',
        'context_reviewed': True, 'task_type': 'pinyin_to_characters',
        'instruction': '按拼音写词语', 'answer_scope': 'all_slots',
        'prompt_regions': [[0, 0, 1, 0.2]],
        'answer_slots': [{'slot_id': 's1', 'bbox': [0, 0.2, 1, 0.8]}],
        'ignore_regions': [], 'reference': {'correct_answer': 'GOLD_MUST_NOT_LEAK'}}]}
    cases_path = tmp_path / 'cases.json'
    cases_path.write_text(json.dumps(cases), encoding='utf-8')
    config = {'content_rounds': 3, 'content_image_max_edge': 2048, 'jpeg_quality': 90,
              'content_timeout_seconds': 20, 'review_outline_width': 2, 'counterbalance_arms': True}
    prepared = prepare(dataset, cases_path, tmp_path / 'prepared', config)
    a, b = prepared['requests']
    assert prepared['planned_http_attempts'] == 6
    assert a['image_sha256'] == b['image_sha256']
    assert a['context_sha256'] == b['context_sha256']
    assert 'GOLD_MUST_NOT_LEAK' not in a['prompt'] + b['prompt']
    assert a['response_schema'] != b['response_schema']
    assert a['expected_slot_ids'] == b['expected_slot_ids'] == {'Q1': ['s1']}


def test_context_boxes_must_be_inside_image():
    from scripts.content_context_experiment import valid_bbox
    with pytest.raises(ValueError): valid_bbox([0, 0, 1.1, 1])


def server_fixture(tmp_path):
    from scripts.prepare_convergence_dataset import sha256
    (tmp_path / 'image.jpg').write_bytes(b'fixture-image')
    requests = []
    for arm, schema in [('A', 'full_content_v2'), ('B', 'transcription_v2')]:
        requests.append({'request_id': arm + '-Q1', 'arm': arm, 'response_schema': schema,
            'annotation_status': 'context-reviewed', 'image': 'image.jpg',
            'image_sha256': sha256(tmp_path / 'image.jpg'), 'context_sha256': 'fixture',
            'expected_ids': ['Q1'], 'expected_slot_ids': {'Q1': ['s1', 's2']},
            'prompt': arm, 'label': 'page'})
    return {'scope': 'content_context_ab', 'dataset_id': 'fixture',
        'expected_questions_per_round': 2,
        'config': {'content_timeout_seconds': 1, 'content_rounds': 3, 'counterbalance_arms': True},
        'requests': requests}


@pytest.mark.parametrize('bad_b', [False, True])
def test_server_counterbalances_and_enforces_arm_schema(tmp_path, monkeypatch, bad_b):
    import httpx
    from app.services.vision_recognition import MiniMaxVisionClient
    from scripts.benchmark_content_oracle import run
    calls = []
    def respond(request):
        arm = json.loads(request.content)['prompt']
        calls.append(arm)
        row = transcription()
        if arm == 'A' or bad_b:
            row.update(correct_answer='冬瓜', error_explanation='fixture')
        return httpx.Response(200, json={'content': json.dumps({'items': [row]})})
    client = MiniMaxVisionClient('fixture-secret', 'https://fixture.invalid', 1, 2, 2048, 90,
                                transport=httpx.MockTransport(respond))
    monkeypatch.setattr(MiniMaxVisionClient, 'from_settings', classmethod(lambda cls: client))
    (tmp_path / 'prepared.json').write_text(json.dumps(server_fixture(tmp_path)), encoding='utf-8')
    report = run(tmp_path, tmp_path / 'out')
    assert calls == ['A', 'B', 'B', 'A', 'A', 'B']
    assert report['http_attempts'] == 6
    for row in report['results']:
        assert row['status'] == ('failed' if row['arm'] == 'B' and bad_b else 'parsed')
    judgments = json.loads((tmp_path / 'out' / 'judgments-template.json').read_text(encoding='utf-8'))
    assert [row['arm'] for row in judgments['judgments']] == calls


@pytest.mark.parametrize('defect', ['schema', 'missing_pair', 'mismatched_image', 'mismatched_slots'])
def test_invalid_ab_package_fails_before_client_creation(tmp_path, monkeypatch, defect):
    from app.services.vision_recognition import MiniMaxVisionClient
    from scripts.benchmark_content_oracle import run
    def forbidden(cls):
        pytest.fail('invalid package must not instantiate a network client')
    monkeypatch.setattr(MiniMaxVisionClient, 'from_settings', classmethod(forbidden))
    prepared = server_fixture(tmp_path)
    if defect == 'schema': prepared['requests'][1]['response_schema'] = 'unknown'
    if defect == 'missing_pair': prepared['requests'].pop()
    if defect == 'mismatched_image': prepared['requests'][1]['image_sha256'] = 'different'
    if defect == 'mismatched_slots': prepared['requests'][1]['expected_slot_ids'] = {'Q1': ['s1']}
    (tmp_path / 'prepared.json').write_text(json.dumps(prepared), encoding='utf-8')
    with pytest.raises(ValueError): run(tmp_path, tmp_path / 'out')
