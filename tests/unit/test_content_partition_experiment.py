import json

from PIL import Image
import pytest

from scripts.content_context_experiment import prepare as prepare_context, ordered_requests, validate_pairs
from scripts.prepare_convergence_dataset import sha256


def fixture_source(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    Image.new('RGB', (160, 120), '#eeeeee').save(dataset / 'page.png')
    manifest = {'dataset_id': 'fixture', 'pages': [{'label': 'page', 'image': 'page.png',
        'image_sha256': sha256(dataset / 'page.png'), 'questions': [{'question_id': 'Q1', 'bbox': [0,0,1,1]}]}]}
    (dataset / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    cases = {'version': 'fixture', 'cases': [{'question_id': 'Q1', 'context_reviewed': True,
        'task_type': 'pinyin_to_characters', 'instruction': '看拼音写词语。', 'answer_scope': 'all_slots',
        'prompt_regions': [[0,0,1,0.2]], 'answer_slots': [{'slot_id': 's1', 'bbox': [0,0.3,0.5,0.8]}],
        'ignore_regions': [[0.7,0.3,1,1]], 'reference': {'status': 'assistant_reviewed',
        'correct_answer': 'GOLD_DO_NOT_SEND'}}]}
    cases_path = tmp_path / 'cases.json'
    cases_path.write_text(json.dumps(cases), encoding='utf-8')
    config = {'content_rounds': 3, 'content_image_max_edge': 2048, 'jpeg_quality': 90,
              'content_timeout_seconds': 20, 'review_outline_width': 2, 'counterbalance_arms': True,
              'margin': 16, 'gap': 16, 'label_height': 26, 'prompt_max_width': 1200,
              'prompt_max_height': 240, 'answer_scale': 2, 'answer_max_edge': 400,
              'smoke_question_id': 'Q1', 'smoke_rounds': 3, 'ocr_max_edge': 1600}
    source = tmp_path / 'source'
    prepare_context(dataset, cases_path, source, config)
    region_path = tmp_path / 'regions.json'
    region_path.write_text(json.dumps({'cases': [{'question_id': 'Q1', 'reviewed': True,
        'slot_cue_regions': {'s1': [0,0,0.5,0.2]}}]}), encoding='utf-8')
    return source, region_path, config


def test_partition_preserves_control_and_excludes_gold(tmp_path):
    from scripts.content_partition_experiment import prepare
    source, regions, config = fixture_source(tmp_path)
    old = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))['requests'][1]
    output = tmp_path / 'prepared'
    result = prepare(source, regions, output, config)
    b, c = result['requests']
    assert b['prompt'] == old['prompt']
    assert (output / b['image']).read_bytes() == (source / old['image']).read_bytes()
    assert b['expected_slot_ids'] == c['expected_slot_ids']
    assert b['response_schema'] == c['response_schema'] == 'transcription_v2'
    assert b['source_context_sha256'] == c['source_context_sha256']
    assert b['image_sha256'] != c['image_sha256']
    assert 'GOLD_DO_NOT_SEND' not in b['prompt'] + c['prompt']
    assert result['planned_http_attempts'] == 6
    assert [r['arm'] for r in ordered_requests(result, 1)] == ['C', 'B']
    validate_pairs(result)
    provenance = json.loads((output / 'partition-provenance.json').read_text(encoding='utf-8'))[0]
    slot = next(t for t in provenance['tiles'] if t['role'] == 'answer')
    with Image.open(source / old['image']) as image, Image.open(output / slot['file']) as crop:
        assert crop.tobytes() == image.crop(tuple(slot['source_pixel_bbox'])).tobytes()
    assert (output / 'smoke' / 'prepared.json').is_file()


def test_missing_slot_cue_is_rejected(tmp_path):
    from scripts.content_partition_experiment import prepare
    source, regions, config = fixture_source(tmp_path)
    regions.write_text(json.dumps({'cases': [{'question_id': 'Q1', 'reviewed': True, 'slot_cue_regions': {}}]}))
    with pytest.raises(ValueError, match='cue'):
        prepare(source, regions, tmp_path / 'prepared', config)


@pytest.mark.parametrize('wrong_c_slot', [False, True])
def test_bc_server_runner_schema_order_and_diagnostics(tmp_path, monkeypatch, wrong_c_slot):
    import httpx
    from app.services.vision_recognition import MiniMaxVisionClient
    from scripts.content_partition_experiment import prepare
    from scripts.benchmark_content_oracle import run
    source, regions, config = fixture_source(tmp_path)
    prepared_dir = tmp_path / 'prepared'
    prepared = prepare(source, regions, prepared_dir, config)
    prompt_arms = {r['prompt']: r['arm'] for r in prepared['requests']}
    calls = []
    def respond(request):
        arm = prompt_arms[json.loads(request.content)['prompt']]
        calls.append(arm)
        return httpx.Response(200, headers={'x-request-id': 'fixture-id'}, json={'content': json.dumps({'items': [{
            'question_id': 'Q1', 'prompt_text': 'fixture', 'answer_slots': [{
                'slot_id': 'wrong' if arm == 'C' and wrong_c_slot else 's1',
                'state': 'blank', 'text': ''}], 'uncertain_segments': []}]})})
    client = MiniMaxVisionClient('fixture-key', 'https://fixture.invalid', 1, 0, 2048, 90,
                                transport=httpx.MockTransport(respond))
    monkeypatch.setattr(MiniMaxVisionClient, 'from_settings', classmethod(lambda cls: client))
    output = tmp_path / 'results'
    report = run(prepared_dir, output)
    assert calls == ['B','C','C','B','B','C']
    assert report['http_attempts'] == 6
    for row in report['results']:
        assert row['status'] == ('failed' if row['arm'] == 'C' and wrong_c_slot else 'parsed')
        assert 'started_at_utc' in row and 'finished_at_utc' in row
    assert 'source_lf_sha256' in report['runtime']
    raw = json.loads((output / 'round1-C-Q1-raw.json').read_text(encoding='utf-8'))
    assert any(e.get('response_ids') == {'x-request-id': 'fixture-id'} for e in raw)
