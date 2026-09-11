import json
from pathlib import Path

import httpx
import pytest
from PIL import Image


def packet(tmp_path):
    from scripts.local_vision_capability import prepare, sha
    image = tmp_path / 'secret-answer.jpg'
    Image.new('RGB', (250, 350), 'white').save(image)
    cases = [dict(source_id='secret-source', source_image=str(image), source_sha256=sha(image),
                  bbox=[.1, .2, .7, .6], group='old_positive', expected='wrong',
                  expected_word='秘密答案', expected_marked_text=None, protected=True)]
    return prepare(cases, Path(__file__).parents[2] / 'scripts/local_vision_capability_config.json', tmp_path/'packet')


def test_contract_does_not_force_transcription_or_promote_empty_detection():
    from scripts.local_vision_capability import Decision, validate_decision
    row = dict(case_id='V001', verdict='wrong', targets=[dict(word=None, marked_text=None)], evidence='red mark')
    validate_decision(Decision(**row), 'V001')
    for bad in [dict(row, targets=[]), dict(row, verdict='not_wrong'), dict(row, case_id='another')]:
        with pytest.raises(ValueError):
            validate_decision(Decision(**bad), 'V001')


def test_packet_never_contains_labels_or_source_names(tmp_path):
    from scripts.local_vision_capability import validate_packet
    folder = packet(tmp_path)
    p = validate_packet(folder)
    text = (folder/'prepared.json').read_text(encoding='utf-8')
    for secret in ['秘密答案', 'secret-source', 'secret-answer', 'old_positive', 'expected_word']:
        assert secret not in text
    assert len(p['schedule']) == 2
    assert folder.with_name('packet-labels.json').exists()


@pytest.mark.parametrize('name', ['images/V001.png', 'local_vision_capability_prompt.md', 'config.json'])
def test_packet_rejects_tampering(tmp_path, name):
    from scripts.local_vision_capability import validate_packet
    folder = packet(tmp_path)
    with (folder/name).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        validate_packet(folder)


def test_target_pixels_are_not_resized(tmp_path):
    from scripts.local_vision_capability import render_input
    im = Image.effect_noise((640, 900), 60).convert('RGB')
    cfg = json.loads((Path(__file__).parents[2]/'scripts/local_vision_capability_config.json').read_text(encoding='utf-8'))
    sheet, metadata = render_input(im, [.1, .2, .7, .6], cfg)
    assert sheet.crop(metadata['target_panel_pixels']).tobytes() == im.crop(metadata['source_crop_pixels']).tobytes()


def test_real_client_transport_failure_is_retained_without_retry(tmp_path):
    from scripts.local_vision_capability import run, read, check_results
    from app.services.vision_recognition import MiniMaxVisionClient
    folder = packet(tmp_path)
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        if len(sent) == 2:
            return httpx.Response(503, json={})
        return httpx.Response(200, json=dict(base_resp=dict(status_code=0), content=json.dumps(
            dict(case_id='V001', verdict='uncertain', targets=[], evidence='unclear'))))
    client = MiniMaxVisionClient(api_key='fake', api_host='https://minimax.invalid', timeout_seconds=20,
                                max_retries=4, max_edge=2048, jpeg_quality=95, transport=httpx.MockTransport(respond))
    run(client, folder, tmp_path/'results', dict(test_only=True))
    assert len(sent) == 2 and sent[0] == sent[1]
    report = read(tmp_path/'results/results.json')
    assert [r['status'] for r in report['results']] == ['parsed', 'failed']
    assert check_results(folder, tmp_path/'results')['integrity_complete']


def test_score_separates_groups_and_does_not_count_unknown_or_duplicates():
    from scripts.local_vision_capability import score_case
    label = dict(expected='wrong', expected_word='秋霜', expected_marked_text='霜')
    row = dict(status='parsed', prediction=dict(verdict='wrong', targets=[dict(word='秋霜', marked_text='霜')]))
    assert score_case(label, row)['target_exact']
    assert not score_case(label, dict(row, prediction=dict(verdict='uncertain', targets=[])))['classification_correct']
    assert not score_case(label, dict(row, prediction=dict(verdict='wrong', targets=row['prediction']['targets']*2)))['target_exact']
    assert not score_case(label, dict(status='failed', prediction=None))['classification_correct']
    negative = dict(expected='not_wrong', expected_word=None, expected_marked_text=None)
    assert not score_case(negative, row)['classification_correct']


def test_partial_run_keeps_missing_old_and_new_cases_in_denominators(tmp_path):
    from scripts.local_vision_capability import prepare, sha, run, score, read
    from app.services.vision_recognition import MiniMaxVisionClient
    image = tmp_path/'image.png'
    Image.new('RGB', (250,350), 'white').save(image)
    common = dict(source_image=str(image), source_sha256=sha(image), expected_word=None, expected_marked_text=None)
    cases = [dict(common, source_id='old', bbox=[0,0,.3,.3], group='old_positive', expected='wrong', protected=True),
             dict(common, source_id='new', bbox=[.5,.5,1,1], group='new_negative', expected='not_wrong', protected=False)]
    folder = prepare(cases, Path(__file__).parents[2]/'scripts/local_vision_capability_config.json', tmp_path/'mixed')
    sent = []
    def respond(request):
        sent.append(request)
        return httpx.Response(503, json={})
    client = MiniMaxVisionClient(api_key='fake', api_host='https://minimax.invalid', timeout_seconds=20,
                                max_retries=4, max_edge=2048, jpeg_quality=95, transport=httpx.MockTransport(respond))
    run(client,folder,tmp_path/'results',dict(test_only=True))
    assert len(sent)==3 and not read(tmp_path/'results/results.json')['complete']
    result = score(folder,tmp_path/'results',tmp_path/'mixed-labels.json')
    assert not result['run_complete']
    for round_ in [1,2]:
        for group in ['old_positive','new_negative','protected']:
            counts = result['groups'][f'{round_}:{group}']
            assert counts['total']==1 and counts['unresolved']==1 and counts['classification_correct']==0


def test_schedule_rejects_budget_overflow_and_duplicate_ids():
    from scripts.local_vision_capability import schedule
    p = dict(config=dict(rounds=2,max_http_attempts=3), requests=[dict(case_id='one'),dict(case_id='two')])
    with pytest.raises(ValueError,match='budget'):
        schedule(p)
    p['config']['max_http_attempts']=4
    p['requests'][1]['case_id']='one'
    with pytest.raises(ValueError,match='duplicate'):
        schedule(p)
