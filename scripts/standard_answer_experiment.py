"""Solve from problem-only input, then compare original slots locally; diagnostic only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import httpx
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_text_oracle import preflight, request_payload
from scripts.content_context_experiment import AnswerSlot
from scripts.experiment_telemetry import error_details, response_ids, runtime_metadata, utc_now
from scripts.prepare_convergence_dataset import sha256, write_json


class StandardSlot(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    slot_id: str
    text: str | None


class StandardAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question_id: str
    standard_slots: list[StandardSlot]
    uncertain_segments: list[str]


def compare_slots(student, prediction, config):
    standard = StandardAnswer.model_validate(prediction)
    original = [AnswerSlot.model_validate(s) for s in student['answer_slots']]
    expected_ids = [s.slot_id for s in original]
    if not original or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('invalid original slot IDs')
    if standard.question_id != student['question_id'] or Counter(s.slot_id for s in standard.standard_slots) != Counter(expected_ids):
        raise ValueError('question or slot ID mismatch')
    separator = config['answer_separators'][student['task_type']]
    normalize = lambda s: unicodedata.normalize('NFC', s).strip()
    lookup = {s.slot_id: s.text for s in standard.standard_slots}
    comparisons, explanations, answers = [], [], []
    unresolved = bool(standard.uncertain_segments)
    for index, slot in enumerate(original, 1):
        expected = lookup[slot.slot_id]
        answers.append(expected)
        if expected is None or not expected.strip() or slot.state == 'uncertain' or slot.text is None:
            status = 'uncertain'
            unresolved = True
        elif slot.state == 'blank':
            status = 'missing'
        elif normalize(slot.text) == normalize(expected):
            status = 'match'
        else:
            status = 'wrong'
        comparisons.append({'slot_id': slot.slot_id, 'original_state': slot.state, 'original_text': slot.text,
                            'standard_text': expected, 'status': status})
        if status in ('wrong', 'missing'):
            explanations.append(config['explanation_templates'][status].format(index=index, expected=expected, actual=slot.text))
    return {'question_id': student['question_id'], 'comparison_complete': not unresolved,
            'question_status': 'needs_review' if unresolved else 'incorrect' if explanations else 'correct',
            'correct_answer': separator.join(answers) if all(a and a.strip() for a in answers) else None,
            'error_explanation': ''.join(explanations) or None, 'slot_comparisons': comparisons}


def prepare(source, cues_path, output, config):
    frozen = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    refs = {c['question_id']: c for c in json.loads((source / 'reference-review.json').read_text(encoding='utf-8'))['cases']}
    selected = json.loads(cues_path.read_text(encoding='utf-8'))['cases']
    if not selected or len({c['question_id'] for c in selected}) != len(selected):
        raise ValueError('empty or duplicate selected questions')
    template = Path(__file__).with_name('standard_answer_prompt.md').read_text(encoding='utf-8')
    requests, students, references = [], [], []
    for entry in selected:
        qid = entry['question_id']
        case = refs[qid]
        reference = case['reference']
        if reference['status'] != 'assistant_reviewed' or case['task_type'] not in config['answer_separators']:
            raise ValueError('unreviewed reference or unsupported task type')
        slots = [AnswerSlot.model_validate(s).model_dump() for s in reference['answer_slots']]
        ids = [s['slot_id'] for s in slots]
        if not ids or len(set(ids)) != len(ids) or set(ids) != set(entry['slot_prompt_texts']):
            raise ValueError('slot cue mismatch')
        public = {key: case[key] for key in ('question_id', 'task_type', 'answer_scope')}
        public.update(instruction=config['instructions'][case['task_type']], prompt_text=reference['prompt_text'],
                      slot_prompt_texts=entry['slot_prompt_texts'])
        schema = StandardAnswer.model_json_schema()
        schema['properties']['question_id']['enum'] = [qid]
        schema['properties']['standard_slots'].update(minItems=len(ids), maxItems=len(ids))
        schema['$defs']['StandardSlot']['properties']['slot_id']['enum'] = ids
        prompt = template + '\n题目输入：\n' + json.dumps(public, ensure_ascii=False)
        prompt += '\n输出JSON Schema：\n' + json.dumps(schema, ensure_ascii=False)
        requests.append({'request_id': qid, 'expected_slot_ids': ids, 'prompt': prompt,
                         'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        students.append({'question_id': qid, 'task_type': case['task_type'], 'answer_slots': slots})
        references.append(case)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'student-inputs.json', {'cases': students})
    write_json(output / 'reference-review.json', {'cases': references})
    prepared = {'scope': 'standard_answer_slots', 'dataset_id': frozen['dataset_id'], 'config': config,
        'requests': requests, 'planned_http_attempts': len(requests) * config['rounds'],
        'student_inputs_sha256': sha256(output / 'student-inputs.json'),
        'reference_sha256': sha256(source / 'reference-review.json'), 'cue_sha256': sha256(cues_path)}
    write_json(output / 'prepared.json', prepared)
    return prepared


def execute_standard(client, payload, student, config):
    started = time.perf_counter()
    result = {'status': 'failed', 'delivery_status': 'failed', 'http_attempts': 1, 'failure_category': None,
              'started_at_utc': utc_now(), 'format_normalization': 'none', 'standard_answer': None, 'comparison': None}
    raw = {}
    try:
        response = client.post('chat/completions', json=payload)
        raw.update(status_code=response.status_code, response_body=response.text)
        result.update(status_code=response.status_code, response_ids=response_ids(response))
        response.raise_for_status()
        data = response.json()
        choice = data['choices'][0]
        result.update(model_returned=data.get('model'), usage=data.get('usage'), finish_reason=choice.get('finish_reason'))
        if result['model_returned'] != payload['model']:
            result['failure_category'] = 'model_mismatch'
            raise ValueError('returned text model differs from requested model')
        if choice.get('finish_reason') == 'length':
            result['failure_category'] = 'output_truncated'
            raise ValueError('truncated final output')
        content = choice['message'].get('content')
        if not isinstance(content, str) or not content.strip():
            result['failure_category'] = 'empty_final_content'
            raise ValueError('missing final content')
        result['final_content_characters'] = len(content)
        candidate = content.strip()
        fence = re.fullmatch(r'```json\s*\n(.*)\n```', candidate, flags=re.DOTALL)
        if config['allow_outer_json_fence'] and fence:
            candidate = fence.group(1)
            result['format_normalization'] = 'outer_json_fence'
        parsed = StandardAnswer.model_validate(json.loads(candidate))
        comparison = compare_slots(student, parsed.model_dump(), config)
        result.update(status='parsed', standard_answer=parsed.model_dump(), comparison=comparison,
                      delivery_status='complete' if comparison['comparison_complete'] else 'incomplete',
                      failure_category=None if comparison['comparison_complete'] else 'uncertain_comparison')
    except Exception as exc:
        result.update(error_type=type(exc).__name__, **error_details(exc))
        if result['failure_category'] is None:
            result['failure_category'] = ('transport_error' if isinstance(exc, httpx.TransportError)
                else 'http_error' if isinstance(exc, httpx.HTTPStatusError) else 'invalid_response')
    result.update(finished_at_utc=utc_now(), elapsed_ms=(time.perf_counter() - started) * 1000)
    return result, raw


def load_prepared(source):
    prepared = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    if prepared['scope'] != 'standard_answer_slots' or not prepared['requests']:
        raise ValueError('invalid standard answer experiment')
    if sha256(source / 'student-inputs.json') != prepared['student_inputs_sha256']:
        raise ValueError('frozen student input hash mismatch')
    students = json.loads((source / 'student-inputs.json').read_text(encoding='utf-8'))['cases']
    ids = [r['request_id'] for r in prepared['requests']]
    if len(set(ids)) != len(ids) or Counter(ids) != Counter(s['question_id'] for s in students):
        raise ValueError('question IDs mismatch')
    lookup = {s['question_id']: s for s in students}
    for request in prepared['requests']:
        if hashlib.sha256(request['prompt'].encode()).hexdigest() != request['prompt_sha256']:
            raise ValueError('frozen prompt hash mismatch')
        student = lookup[request['request_id']]
        if student['task_type'] not in prepared['config']['answer_separators']:
            raise ValueError('unsupported comparison task')
        if Counter(request['expected_slot_ids']) != Counter(s['slot_id'] for s in student['answer_slots']):
            raise ValueError('frozen original slot IDs mismatch')
    return prepared, lookup


def run(source, output):
    from app.config import settings
    prepared, students = load_prepared(source)
    identity = preflight(prepared['config'])
    if not settings.LLM_API_KEY:
        raise ValueError('text API key is not configured')
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': prepared['scope'], 'prepared_sha256': sha256(source / 'prepared.json'),
        'student_inputs_sha256': prepared['student_inputs_sha256'], 'service_identity': identity,
        'started_at_utc': utc_now(), 'complete': False, 'results': [],
        'runtime': runtime_metadata(settings.LLM_API_BASE, ['scripts/standard_answer_experiment.py',
            'scripts/benchmark_text_oracle.py', 'scripts/content_context_experiment.py',
            'scripts/experiment_telemetry.py', 'app/config.py'], settings.LLM_MODEL)}
    config = prepared['config']
    with httpx.Client(base_url=settings.LLM_API_BASE.rstrip('/') + '/', timeout=config['timeout_seconds'],
                      headers={'Authorization': 'Bearer ' + settings.LLM_API_KEY}) as client:
        for n in range(1, config['rounds'] + 1):
            for request in prepared['requests']:
                payload = request_payload(request['prompt'], config, settings.LLM_MODEL)
                result, raw = execute_standard(client, payload, students[request['request_id']], config)
                result.update(round=n, request_id=request['request_id'])
                write_json(output / f"round{n}-{request['request_id']}-raw.json", raw)
                report['results'].append(result)
                write_json(output / 'results.json', report)
                print(json.dumps({k: result[k] for k in ('round', 'request_id', 'status', 'delivery_status', 'elapsed_ms')}), flush=True)
    report.update(complete=True, finished_at_utc=utc_now(), http_attempts=len(report['results']),
        delivery_counts=dict(Counter(r['delivery_status'] for r in report['results'])),
        normalization_counts=dict(Counter(r['format_normalization'] for r in report['results'])))
    write_json(output / 'results.json', report)
    write_json(output / 'judgments-template.json', {'judgments': [{'round': r['round'], 'question_id': r['request_id'],
        'standard_answer': r['standard_answer'], 'comparison': r['comparison'], 'reviewed': False,
        'standard_slots_correct': None, 'all_errors_explained': None} for r in report['results']]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'preflight', 'run'], default='prepare')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('standard_answer_config.json'))
    parser.add_argument('--cues', type=Path, default=Path(__file__).with_name('text_oracle_inputs.json'))
    args = parser.parse_args()
    if args.mode == 'preflight':
        prepared, _ = load_prepared(args.source)
        print(json.dumps(preflight(prepared['config']), ensure_ascii=False, indent=2))
    elif args.output is None:
        parser.error('--output required for prepare/run')
    elif args.mode == 'run':
        run(args.source, args.output)
    else:
        prepared = prepare(args.source, args.cues, args.output, json.loads(args.config.read_text(encoding='utf-8')))
        print(json.dumps({'questions': len(prepared['requests']), 'planned_http_attempts': prepared['planned_http_attempts']}))


if __name__ == '__main__':
    main()
