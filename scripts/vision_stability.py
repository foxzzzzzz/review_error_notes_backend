"""One bounded three-window study around the unchanged blind-original runner."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import blind_original_experiment as blind
from scripts.experiment_telemetry import utc_now, runtime_metadata
from scripts.prepare_convergence_dataset import sha256, write_json


def prepare(source, output, config):
    if sha256(source / 'prepared.json') != config['baseline_prepared_sha256']:
        raise ValueError('baseline prepared hash mismatch')
    if sha256(source / 'reference-review.json') != config['reference_sha256']:
        raise ValueError('reference hash mismatch')
    baseline = blind.load_prepared(source)
    if len(baseline['requests']) != config['questions']:
        raise ValueError('question count mismatch')
    output.mkdir(parents=True, exist_ok=False)
    for r in baseline['requests']:
        shutil.copyfile(source / r['image'], output / r['image'])
    shutil.copyfile(source / 'reference-review.json', output / 'reference-review.json')
    shutil.copyfile(source / 'prepared.json', output / 'baseline-prepared.json')
    prepared = deepcopy(baseline)
    prepared['config']['rounds'] = config['repeats_per_window']
    prepared['planned_http_attempts'] = config['questions'] * config['repeats_per_window']
    write_json(output / 'prepared.json', prepared)
    write_json(output / 'stability-config.json', config)
    validate_source(output)
    return prepared


def validate_source(source):
    cfg = json.loads((source / 'stability-config.json').read_text(encoding='utf-8'))
    if sha256(source / 'baseline-prepared.json') != cfg['baseline_prepared_sha256']:
        raise ValueError('baseline prepared hash mismatch')
    if sha256(source / 'reference-review.json') != cfg['reference_sha256']:
        raise ValueError('reference hash mismatch')
    expected = json.loads((source / 'baseline-prepared.json').read_text(encoding='utf-8'))
    expected['config']['rounds'] = cfg['repeats_per_window']
    expected['planned_http_attempts'] = cfg['questions'] * cfg['repeats_per_window']
    actual = blind.load_prepared(source)
    if actual != expected or len(actual['requests']) != cfg['questions']:
        raise ValueError('images/prompts/request configuration changed')
    if (cfg['repeats_per_window'] != 2 or not cfg['windows'] or
            len(set(cfg['windows'])) != len(cfg['windows']) or
            len(cfg['windows']) * actual['planned_http_attempts'] != cfg['max_http_attempts']):
        raise ValueError('invalid study budget')
    return cfg, actual


def check_window(ledger, config, window, now):
    planned = config['windows']
    done = ledger['windows']
    if window in done:
        raise ValueError('window already reserved; no overwrite or rerun')
    if window not in planned or list(done) != planned[:len(done)] or window != planned[len(done)]:
        raise ValueError('window order mismatch')
    reservation = config['questions'] * config['repeats_per_window']
    if sum(v['reserved_attempts'] for v in done.values()) + reservation > config['max_http_attempts']:
        raise ValueError('study request budget exhausted')
    if done:
        previous = done[planned[len(done) - 1]]
        if previous['status'] not in ('complete', 'aborted'):
            raise ValueError('previous window is still running')
        delta = (now - datetime.fromisoformat(previous['started_at_utc'])).total_seconds()
        if delta < config['minimum_start_gap_seconds']:
            raise ValueError('minimum start gap not reached')


def active_counts(replies):
    if not replies or not isinstance(replies, dict) or not all(isinstance(v, list) for v in replies.values()):
        return {'responding_workers': 0, 'active_tasks': None}
    return {'responding_workers': len(replies), 'active_tasks': sum(len(v) for v in replies.values())}


def environment_sample(config):
    sample = {'sample_started_at_utc': utc_now(), 'minimax_actual_concurrency': None,
              'responding_workers': 0, 'active_tasks': None}
    try:
        sample['host_load_average'] = list(os.getloadavg())
    except (AttributeError, OSError):
        sample['host_load_average'] = None
    # Read-only inspect in a bounded subprocess. Discard task IDs, names, args and all stderr.
    code = ('import json; from app.tasks.celery_app import celery_app; '
            'from scripts.vision_stability import active_counts; '
            'print(json.dumps(active_counts(celery_app.control.inspect(timeout=' + str(config['inspect_timeout_seconds']) + ').active())))')
    try:
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                                timeout=config['sample_process_timeout_seconds'], check=True)
        counts = json.loads(result.stdout.strip().splitlines()[-1])
        sample.update({k: counts[k] for k in ('responding_workers', 'active_tasks')})
        sample['status'] = 'observed' if counts['active_tasks'] is not None else 'no_worker_reply'
    except Exception as exc:
        sample.update(status='unavailable', error_type=type(exc).__name__)
    sample['sample_finished_at_utc'] = utc_now()
    return sample


def run_window(source, campaign, window, load_note, sampler=None):
    config, prepared = validate_source(source)
    campaign.mkdir(parents=True, exist_ok=True)
    lock = campaign / '.run-lock'
    lock.mkdir(exist_ok=False)
    try:
        binding = {'study_id': config['study_id'], 'prepared_sha256': sha256(source / 'prepared.json'),
                   'config_sha256': sha256(source / 'stability-config.json')}
        path = campaign / 'campaign.json'
        ledger = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(binding, windows={})
        if any(ledger[k] != v for k, v in binding.items()):
            raise ValueError('campaign input changed')
        check_window(ledger, config, window, datetime.now(timezone.utc))
        blind.checked_client(prepared['config'])  # Identity/key check only; no image request.
        folder = campaign / window
        folder.mkdir(exist_ok=False)
        record = {'status': 'running', 'started_at_utc': utc_now(),
                  'reserved_attempts': prepared['planned_http_attempts'], 'load_note': load_note,
                  'experiment_request_concurrency': 1, 'other_minimax_clients': 'not_observed',
                  'observer_source': runtime_metadata(None, ['scripts/vision_stability.py'])}
        ledger['windows'][window] = record
        write_json(path, ledger)  # Reserve all 20 calls before the first request, even if later aborted.
        stop = threading.Event()
        errors = []
        sampler = sampler or environment_sample
        def monitor():
            try:
                while True:
                    sample = sampler(config)
                    with (folder / 'environment.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    if stop.wait(config['sample_interval_seconds']):
                        break
            except Exception as exc:
                errors.append(type(exc).__name__)
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        try:
            report = blind.run(source, folder / 'results')
            record.update(status='complete', http_attempts=report['http_attempts'])
        except BaseException:
            record['status'] = 'aborted'
            raise
        finally:
            stop.set()
            worker.join(timeout=config['sample_process_timeout_seconds'] + config['sample_interval_seconds'])
            record.update(finished_at_utc=utc_now(), observer_stopped=not worker.is_alive(), observer_errors=errors)
            write_json(path, ledger)
        return ledger
    finally:
        lock.rmdir()


def retry_choice(first, second, triggers):
    retry = first['delivery_status'] in triggers
    return (second if retry else first, (first['elapsed_ms'] + (second['elapsed_ms'] if retry else 0)) / 1000, retry)


def grade(row, reference):
    if row['status'] != 'parsed' or len(row['items']) != 1:
        return {'strict_pass': False, 'literal_match': False, 'semantic_wrong': False}
    item = row['items'][0]
    expected = {s['slot_id']: s for s in reference['answer_slots']}
    observed = {s['slot_id']: s for s in item['answer_slots']}
    literal = item['question_id'] == reference['question_id'] and len(observed) == len(item['answer_slots']) and observed == expected
    wrong = any(s['state'] != 'uncertain' and s != expected.get(sid) for sid, s in observed.items())
    return {'strict_pass': literal and row['delivery_status'] == 'complete', 'literal_match': literal, 'semantic_wrong': wrong}


def summarize_environment(path):
    samples = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    counts = [s['active_tasks'] for s in samples if s.get('active_tasks') is not None]
    loads = [s['host_load_average'][0] for s in samples if s.get('host_load_average')]
    return {'samples': len(samples), 'active_tasks_observed_samples': len(counts),
            'active_tasks_min': min(counts) if counts else None,
            'active_tasks_max': max(counts) if counts else None,
            'host_load_1m_min': min(loads) if loads else None,
            'host_load_1m_max': max(loads) if loads else None,
            'minimax_actual_concurrency': None}


def summarize(source, campaign):
    config, prepared = validate_source(source)
    ledger = json.loads((campaign / 'campaign.json').read_text(encoding='utf-8'))
    if ledger['config_sha256'] != sha256(source / 'stability-config.json') or ledger['prepared_sha256'] != sha256(source / 'prepared.json'):
        raise ValueError('campaign binding mismatch')
    cases = json.loads((source / 'reference-review.json').read_text(encoding='utf-8'))['cases']
    references = {r['question_id']: r for r in cases}
    if len(references) != len(cases) or set(references) != {r['request_id'] for r in prepared['requests']}:
        raise ValueError('reference question mismatch')
    for request in prepared['requests']:
        slots = references[request['request_id']]['answer_slots']
        if len(slots) != len({s['slot_id'] for s in slots}) or {s['slot_id'] for s in slots} != set(request['expected_slot_ids'][request['request_id']]):
            raise ValueError('reference slot mismatch')
    windows, rows, replay = {}, [], []
    for window in config['windows']:
        entry = ledger['windows'].get(window)
        if not entry or entry['status'] != 'complete':
            raise ValueError('study incomplete; preserve partial data, do not rerun reserved windows')
        if entry['reserved_attempts'] != prepared['planned_http_attempts'] or entry['http_attempts'] != prepared['planned_http_attempts']:
            raise ValueError('ledger budget mismatch')
        folder = campaign / window / 'results'
        result = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
        expected_pairs = {(q, n) for q in references for n in range(1, config['repeats_per_window'] + 1)}
        if not result['complete'] or result['prepared_sha256'] != ledger['prepared_sha256'] or result['http_attempts'] != prepared['planned_http_attempts']:
            raise ValueError('window incomplete or HTTP count mismatch')
        if len(result['results']) != len(expected_pairs) or {(r['request_id'], r['round']) for r in result['results']} != expected_pairs:
            raise ValueError('missing or duplicate observations')
        if len(list(folder.glob('*-raw.json'))) != len(expected_pairs):
            raise ValueError('raw count mismatch')
        transports = Counter()
        for r in result['results']:
            raw = json.loads((folder / f"round{r['round']}-{r['request_id']}-raw.json").read_text(encoding='utf-8'))
            if sum(e['kind'] == 'request' for e in raw) != 1 or r['http_attempts'] != 1:
                raise ValueError('unexpected retry')
            transports.update(e.get('transport_error_type') or 'unknown' for e in raw if e['kind'] == 'transport_error')
            rows.append(dict(r, window=window, **grade(r, references[r['request_id']])))
        group = [r for r in rows if r['window'] == window]
        secs = [r['elapsed_ms'] / 1000 for r in group]
        windows[window] = {'strict_pass': sum(r['strict_pass'] for r in group), 'total': len(group),
                           'first_pass': sum(r['strict_pass'] for r in group if r['round'] == 1),
                           'semantic_wrong': sum(r['semantic_wrong'] for r in group),
                           'delivery_counts': dict(Counter(r['delivery_status'] for r in group)),
                           'transport_error_counts': dict(transports),
                           'seconds_mean': statistics.mean(secs), 'seconds_max': max(secs),
                           'load_note': entry['load_note'],
                           'environment': summarize_environment(campaign / window / 'environment.jsonl')}
        for qid in references:
            first, second = sorted((r for r in group if r['request_id'] == qid), key=lambda r: r['round'])
            selected, seconds, retried = retry_choice(first, second, config['retry_replay_triggers'])
            replay.append({'window': window, 'question_id': qid, 'selected_round': selected['round'],
                           'retried': retried, 'charged_request_seconds': seconds, **grade(selected, references[qid])})
    questions = {}
    for qid in references:
        group = [r for r in rows if r['request_id'] == qid]
        fingerprints = {json.dumps(sorted(r['items'][0]['answer_slots'], key=lambda s: s['slot_id']), ensure_ascii=False, sort_keys=True)
                        for r in group if r['status'] == 'parsed'}
        questions[qid] = {'strict_pass': sum(r['strict_pass'] for r in group), 'total': len(group),
                          'semantic_wrong': sum(r['semantic_wrong'] for r in group), 'distinct_parsed_slot_outputs': len(fingerprints)}
    passing = sum(r['strict_pass'] for r in rows)
    gate = (passing >= math.ceil(config['joint_accuracy_min'] * config['max_http_attempts'])
            and all(v['strict_pass'] >= math.ceil(config['joint_accuracy_min'] * v['total']) for v in windows.values())
            and all(v['semantic_wrong'] <= config['max_semantic_failures_per_question'] for v in questions.values()))
    summary = {'study_id': config['study_id'], 'complete': True, 'http_attempts': len(rows),
               'strict_pass': passing, 'windows': windows, 'questions': questions,
               'first_pass': sum(r['strict_pass'] for r in rows if r['round'] == 1),
               'retry_replay': {'strict_pass': sum(r['strict_pass'] for r in replay), 'total': len(replay),
                   'triggered': sum(r['retried'] for r in replay),
                   'single_question_over_page_budget': sum(r['charged_request_seconds'] > config['page_processing_limit_seconds'] for r in replay),
                   'note': 'Offline replay with fixed visible-error triggers; not real immediate retries or measured page latency.'},
               'decision': 'candidate_stability_gate_passed_not_production_closed' if gate else 'close_current_protocol_not_ready',
               'no_additional_calls_planned': True, 'environment_note': 'Celery active tasks are a sampled proxy; MiniMax global concurrency is unknown.'}
    write_json(campaign / 'summary.json', summary)
    write_json(campaign / 'judgments.json', rows)
    write_json(campaign / 'retry-replay.json', replay)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'preflight', 'run-window', 'summarize'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--window')
    parser.add_argument('--load-note', default='Other concurrent MiniMax clients have not been confirmed.')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('vision_stability_config.json'))
    args = parser.parse_args()
    if args.mode == 'preflight':
        cfg, p = validate_source(args.source)
        blind.checked_client(p['config'])
        result = {'study_id': cfg['study_id'], 'windows': cfg['windows'], 'max_http_attempts': cfg['max_http_attempts'],
                  'minimum_start_gap_seconds': cfg['minimum_start_gap_seconds'], 'real_network_calls': 0}
    elif args.output is None:
        parser.error('--output required')
    elif args.mode == 'prepare':
        result = prepare(args.source, args.output, json.loads(args.config.read_text(encoding='utf-8')))
        result = {'questions': len(result['requests']), 'per_window_attempts': result['planned_http_attempts'], 'real_network_calls': 0}
    elif args.mode == 'run-window':
        if args.window is None:
            parser.error('--window required')
        result = run_window(args.source, args.output, args.window, args.load_note)
    else:
        result = summarize(args.source, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
