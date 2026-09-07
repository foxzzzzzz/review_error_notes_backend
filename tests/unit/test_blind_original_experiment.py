import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.prepare_convergence_dataset import sha256, write_json


def test_blind_input_contains_only_answer_pixels_and_no_reference(tmp_path):
    from scripts.blind_original_experiment import prepare, load_prepared
    Image.new('RGB', (31, 25), 'gray').save(tmp_path / 'answer.png')
    Image.new('RGB', (31, 25), 'red').save(tmp_path / 'cue.png')
    write_json(tmp_path / 'prepared.json', {'scope': 'known_slot_ocr', 'tiles': [
        {'question_id': 'Q', 'slot_id': 's1', 'role': role, 'file': file, 'sha256': sha256(tmp_path / file),
         'reference_text': reference} for role, file, reference in [('answer', 'answer.png', '原答参考'), ('cue', 'cue.png', '题面参考')]]})
    config = {'rounds': 3, 'margin': 10, 'gap': 10, 'label_height': 24, 'max_edge': 2048,
              'timeout_seconds': 20, 'expected_endpoint_host': 'api.minimaxi.com'}
    out = tmp_path / 'out'
    r = prepare([tmp_path], out, config)
    assert r['planned_http_attempts'] == 3
    assert '原答参考' not in json.dumps(r, ensure_ascii=False)
    assert '题面参考' not in json.dumps(r, ensure_ascii=False)
    req = r['requests'][0]
    with Image.open(out / req['image']) as image:
        assert image.crop((10, 34, 41, 59)).tobytes() == Image.open(tmp_path / 'answer.png').tobytes()
    assert load_prepared(out) == r
    (out / req['image']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='image hash'):
        load_prepared(out)


def test_blind_schema_rejects_duplicate_slots_and_preserves_uncertain():
    from scripts.blind_original_experiment import OriginalResult
    from scripts.benchmark_content_oracle import execute_request
    item = {'question_id': 'Q', 'answer_slots': [
        {'slot_id': 's1', 'state': 'uncertain', 'text': None}], 'uncertain_segments': ['笔迹不清']}
    client = SimpleNamespace(_request=lambda *_: OriginalResult.model_validate({'items': [item]}))
    r = execute_request(client, {}, ['Q'], OriginalResult, {'Q': ['s1']})
    assert r['status'] == 'parsed' and r['items'][0]['answer_slots'][0]['state'] == 'uncertain'
    item['answer_slots'].append(dict(item['answer_slots'][0]))
    r = execute_request(client, {}, ['Q'], OriginalResult, {'Q': ['s1']})
    assert r['status'] == 'failed'


def test_full_run_preserves_raw_and_failure_without_reading_reference(tmp_path, monkeypatch):
    from scripts import blind_original_experiment as blind
    import hashlib
    Image.new('RGB', (20, 20), 'white').save(tmp_path / 'image.png')
    prompt = 'public-only'
    write_json(tmp_path / 'prepared.json', {'scope': 'blind_original_slots', 'config': {'rounds': 2},
        'requests': [{'request_id': 'Q', 'expected_ids': ['Q'], 'expected_slot_ids': {'Q': ['s1']},
                     'image': 'image.png', 'image_sha256': sha256(tmp_path / 'image.png'),
                     'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}]})
    class FakeClient:
        api_host = 'https://api.minimaxi.com'
        count = 0
        _post = lambda *args: None
        def _request(self, payload, model, context):
            self.count += 1
            self.diagnostic_event_sink({'kind': 'request', 'attempt': 1})
            self.diagnostic_event_sink({'kind': 'response', 'status_code': 200, 'response_body': 'retained'})
            if self.count == 2:
                raise ValueError('bad response')
            return model.model_validate({'items': [{'question_id': 'Q', 'answer_slots': [
                {'slot_id': 's1', 'state': 'blank', 'text': ''}], 'uncertain_segments': []}]})
    client = FakeClient()
    monkeypatch.setattr(blind, 'checked_client', lambda _: client)
    result = blind.run(tmp_path, tmp_path / 'results')
    assert result['complete'] and result['http_attempts'] == client.count == 2
    assert result['delivery_counts'] == {'complete': 1, 'failed': 1}
    assert len(list((tmp_path / 'results').glob('*-raw.json'))) == 2
    assert not (tmp_path / 'reference-review.json').exists()
