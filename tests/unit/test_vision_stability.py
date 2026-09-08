import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_convergence_dataset import sha256, write_json


def config():
    return {'windows': ['w1', 'w2', 'w3'], 'questions': 10, 'repeats_per_window': 2,
            'max_http_attempts': 60, 'minimum_start_gap_seconds': 3600}


def test_window_budget_order_and_spacing_are_enforced():
    from scripts.vision_stability import check_window
    cfg = config()
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    ledger = {'windows': {}}
    with pytest.raises(ValueError, match='order'):
        check_window(ledger, cfg, 'w2', now)
    check_window(ledger, cfg, 'w1', now)
    ledger['windows']['w1'] = {'started_at_utc': now.isoformat(), 'reserved_attempts': 20, 'status': 'complete'}
    with pytest.raises(ValueError, match='already reserved'):
        check_window(ledger, cfg, 'w1', now)
    with pytest.raises(ValueError, match='gap'):
        check_window(ledger, cfg, 'w2', now + timedelta(seconds=3599))
    check_window(ledger, cfg, 'w2', now + timedelta(hours=1))
    ledger['windows']['w2'] = {'started_at_utc': (now + timedelta(hours=1)).isoformat(),
                             'reserved_attempts': 20, 'status': 'aborted'}
    check_window(ledger, cfg, 'w3', now + timedelta(hours=2))
    ledger['windows']['w3'] = {'started_at_utc': (now + timedelta(hours=2)).isoformat(),
                             'reserved_attempts': 20, 'status': 'complete'}
    with pytest.raises(ValueError):
        check_window(ledger, cfg, 'w4', now + timedelta(hours=3))
    ledger['windows'].pop('w3')
    ledger['windows']['w2']['reserved_attempts'] = 40
    with pytest.raises(ValueError, match='budget'):
        check_window(ledger, cfg, 'w3', now + timedelta(hours=2))


def test_retry_selection_uses_delivery_only_never_reference():
    from scripts.vision_stability import retry_choice
    first = {'delivery_status': 'complete', 'items': [{'wrong': 'but complete'}], 'elapsed_ms': 4000}
    second = {'delivery_status': 'complete', 'items': [{'correct': 'by chance'}], 'elapsed_ms': 5000}
    selected, seconds, retried = retry_choice(first, second, ['failed', 'incomplete'])
    assert selected is first and seconds == 4 and retried is False
    first = copy.deepcopy(first)
    first['delivery_status'] = 'failed'
    selected, seconds, retried = retry_choice(first, second, ['failed', 'incomplete'])
    assert selected is second and seconds == 9 and retried is True


def test_unknown_worker_replies_are_not_zero_concurrency():
    from scripts.vision_stability import active_counts
    assert active_counts(None)['active_tasks'] is None
    assert active_counts({})['active_tasks'] is None
    counts = active_counts({'worker': [{'args': 'must not be exported'}], 'idle': []})
    assert counts == {'responding_workers': 2, 'active_tasks': 1}
    assert active_counts({'idle': []})['active_tasks'] == 0


@pytest.fixture
def study(tmp_path):
    from scripts import vision_stability as study
    source = tmp_path / 'old'
    source.mkdir()
    Image.new('RGB', (20, 20), 'white').save(source / 'image.png')
    write_json(source / 'prepared.json', {'scope': 'blind_original_slots', 'config': {'rounds': 3},
        'planned_http_attempts': 3, 'requests': [{'request_id': 'Q', 'expected_ids': ['Q'],
        'expected_slot_ids': {'Q': ['s1']}, 'image': 'image.png', 'image_sha256': sha256(source / 'image.png'),
        'prompt': 'public-only', 'prompt_sha256': hashlib.sha256(b'public-only').hexdigest()}]})
    write_json(source / 'reference-review.json', {'cases': [{'question_id': 'Q', 'answer_slots': [
        {'slot_id': 's1', 'state': 'blank', 'text': ''}]}]})
    cfg = json.loads(Path(study.__file__).with_name('vision_stability_config.json').read_text(encoding='utf-8'))
    cfg.update(questions=1, max_http_attempts=6, minimum_start_gap_seconds=0,
               baseline_prepared_sha256=sha256(source / 'prepared.json'),
               reference_sha256=sha256(source / 'reference-review.json'))
    output = tmp_path / 'input'
    prepared = study.prepare(source, output, cfg)
    assert prepared['requests'] == json.loads((source / 'prepared.json').read_text())['requests']
    assert sha256(source / 'image.png') == sha256(output / 'image.png')
    return output, tmp_path / 'campaign'


def test_reference_tampering_rejected_before_calls(study):
    from scripts.vision_stability import validate_source
    source, _ = study
    (source / 'reference-review.json').write_text('{}')
    with pytest.raises(ValueError, match='reference hash'):
        validate_source(source)


@pytest.mark.parametrize('wrong_first', [True, False])
def test_real_runner_fake_client_binds_windows_and_never_picks_best(study, monkeypatch, wrong_first):
    from scripts import vision_stability as mod
    source, campaign = study
    class FakeClient:
        api_host = 'https://api.minimaxi.com'
        count = 0
        _post = lambda *args: None
        def _request(self, payload, model, context):
            self.count += 1
            self.diagnostic_event_sink({'kind': 'request', 'attempt': 1})
            assert 'reference' not in payload['prompt']
            return model.model_validate({'items': [{'question_id': 'Q', 'answer_slots': [
                {'slot_id': 's1', 'state': 'written', 'text': '错'} if wrong_first and self.count % 2 else
                {'slot_id': 's1', 'state': 'blank', 'text': ''}], 'uncertain_segments': []}]})
    client = FakeClient()
    monkeypatch.setattr(mod.blind, 'checked_client', lambda _: client)
    for window in ('w1', 'w2', 'w3'):
        mod.run_window(source, campaign, window, 'unknown', sampler=lambda _: {'active_tasks': None})
        with pytest.raises(ValueError, match='already reserved'):
            mod.run_window(source, campaign, window, 'unknown')
    result = mod.summarize(source, campaign)
    assert client.count == result['http_attempts'] == 6
    assert result['strict_pass'] == (3 if wrong_first else 6)
    assert result['retry_replay']['strict_pass'] == (0 if wrong_first else 3)
    assert result['retry_replay']['triggered'] == 0
    assert result['decision'] == ('close_current_protocol_not_ready' if wrong_first else 'candidate_stability_gate_passed_not_production_closed')
    assert result['windows']['w1']['environment']['active_tasks_max'] is None
    assert len(list(campaign.glob('*/results/*-raw.json'))) == 6
    raw = campaign / 'w1/results/round1-Q-raw.json'
    write_json(raw, [{'kind': 'request'}, {'kind': 'request'}])
    with pytest.raises(ValueError, match='unexpected retry'):
        mod.summarize(source, campaign)


def test_aborted_window_keeps_reservation_and_cannot_be_rerun(study, monkeypatch):
    from scripts import vision_stability as mod
    source, campaign = study
    monkeypatch.setattr(mod.blind, 'checked_client', lambda _: None)
    def abort(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(mod.blind, 'run', abort)
    with pytest.raises(KeyboardInterrupt):
        mod.run_window(source, campaign, 'w1', 'unknown', sampler=lambda _: {'active_tasks': None})
    ledger = json.loads((campaign / 'campaign.json').read_text())
    assert ledger['windows']['w1']['status'] == 'aborted'
    assert ledger['windows']['w1']['reserved_attempts'] == 2
    with pytest.raises(ValueError, match='already reserved'):
        mod.run_window(source, campaign, 'w1', 'unknown')
    with pytest.raises(ValueError, match='incomplete'):
        mod.summarize(source, campaign)


def test_environment_summary_keeps_missing_samples_distinct_from_idle(tmp_path):
    from scripts.vision_stability import summarize_environment
    path = tmp_path / 'environment.jsonl'
    assert summarize_environment(path)['active_tasks_min'] is None
    path.write_text('\n'.join(json.dumps(s) for s in [
        {'active_tasks': None}, {'active_tasks': 0, 'host_load_average': [1, 2, 3]},
        {'active_tasks': 4, 'host_load_average': [2, 3, 4]}]), encoding='utf-8')
    r = summarize_environment(path)
    assert r['samples'] == 3 and r['active_tasks_observed_samples'] == 2
    assert r['active_tasks_min'] == 0 and r['active_tasks_max'] == 4
    assert r['minimax_actual_concurrency'] is None
